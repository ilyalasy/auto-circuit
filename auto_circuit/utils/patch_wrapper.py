from datetime import datetime
from typing import Any, Optional, Tuple

import torch as t
from einops import einsum

from auto_circuit.types import MaskFn, PatchWrapper
from auto_circuit.utils.tensor_ops import assign_sparse_tensor, sample_hard_concrete

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


class PatchWrapperImpl(PatchWrapper):
    """
    PyTorch module that wraps another module, a [`Node`][auto_circuit.types.Node] in
    the computation graph of the model. Implements the abstract
    [`PatchWrapper`][auto_circuit.types.PatchWrapper] class, which exists to work around
    circular import issues.

    If the wrapped module is a [`SrcNode`][auto_circuit.types.SrcNode], the tensor
    `self.curr_src_outs` (a single instance of which is shared by all PatchWrappers in
    the model) is updated with the output of the wrapped module.

    If the wrapped module is a [`DestNode`][auto_circuit.types.DestNode], the input to
    the wrapped module is adjusted in order to interpolate the activations of the
    incoming edges between the default activations (`self.curr_src_outs`) and the
    ablated activations (`self.patch_src_outs`).

    Note:
        Most `PatchWrapper`s are both [`SrcNode`][auto_circuit.types.SrcNode]s and
        [`DestNode`][auto_circuit.types.DestNode]s.

    Args:
        module_name: Name of the wrapped module.
        module: The module to wrap.
        head_dim: The dimension along which to split the heads. In TransformerLens
            `HookedTransformer`s this is `2` because the activations have shape
            `[batch, seq_len, n_heads, head_dim]`.
        seq_dim: The sequence dimension of the model. This is the dimension on which new
            inputs are concatenated. In transformers, this is `1` because the
            activations are of shape `[batch_size, seq_len, hidden_dim]`.
        is_src: Whether the wrapped module is a [`SrcNode`][auto_circuit.types.SrcNode].
        src_idxs: The slice of the list of indices of
            [`SrcNode`][auto_circuit.types.SrcNode]s which output from this
            module. This is used to slice the shared `curr_src_outs` tensor when
            updating the activations of the current forward pass.
        is_dest (bool): Whether the wrapped module is a
            [`DestNode`][auto_circuit.types.DestNode].
        patch_mask: The mask that interpolates between the default activations
            (`curr_src_outs`) and the ablation activations (`patch_src_outs`).
        in_srcs: The slice of the list of indices of
            [`SrcNode`][auto_circuit.types.SrcNode]s which input to this module. This is
            used to slice the shared `curr_src_outs` tensor and the shared
            `patch_src_outs` tensor, when interpolating the activations of the incoming
            edges.
    """

    def __init__(
        self,
        module_name: str,
        module: t.nn.Module,
        head_dim: Optional[int] = None,
        seq_dim: Optional[int] = None,
        is_src: bool = False,
        src_idxs: Optional[slice] = None,
        is_dest: bool = False,
        patch_mask: Optional[t.Tensor] = None,
        in_srcs: Optional[slice] = None,
    ):
        super().__init__()
        self.module_name: str = module_name
        self.module: t.nn.Module = module
        self.head_dim: Optional[int] = head_dim
        self.seq_dim: Optional[int] = seq_dim
        self.curr_src_outs: Optional[t.Tensor] = None
        self.in_srcs: Optional[slice] = in_srcs

        self.is_src = is_src
        if self.is_src:
            assert src_idxs is not None
            self.src_idxs: slice = src_idxs

        self.is_dest = is_dest
        if self.is_dest:
            assert patch_mask is not None
            self.patch_mask: t.nn.Parameter = t.nn.Parameter(patch_mask)
            self.patch_src_outs: Optional[t.Tensor] = None
            self.mask_fn: MaskFn = None
            self.dropout_layer: t.nn.Module = t.nn.Dropout(p=0.0)
        self.patch_mode = False
        self.batch_size = None

        assert head_dim is None or seq_dim is None or head_dim > seq_dim
        dims = range(1, max(head_dim if head_dim else 2, seq_dim if seq_dim else 2))
        self.dims = " ".join(["seq" if i == seq_dim else f"d{i}" for i in dims])

    def set_mask_batch_size(self, batch_size: int | None):
        """
        Set the batch size of the patch mask. Should only be used by context manager
        [`set_mask_batch_size`][auto_circuit.utils.graph_utils.set_mask_batch_size]

        The current primary use case is to collect gradients on the patch mask for
        each input in the batch.

        Warning:
            This is an experimental feature that breaks some parts of the library and
            should be used with caution.

        Args:
            batch_size: The batch size of the patch mask.
        """
        if batch_size is None and self.batch_size is None:
            return
        if batch_size is None:  # removing batch dim
            self.patch_mask = t.nn.Parameter(self.patch_mask[0].clone())
        elif self.batch_size is None:  # adding batch_dim
            self.patch_mask = t.nn.Parameter(
                self.patch_mask.repeat(batch_size, *((1,) * self.patch_mask.ndim))
            )
        elif self.batch_size != batch_size:  # modifying batch dim
            self.patch_mask = t.nn.Parameter(
                self.patch_mask[0]
                .clone()
                .repeat(batch_size, *((1,) * self.patch_mask.ndim))
            )
        self.batch_size = batch_size

    def forward(self, *args: Any, **kwargs: Any) -> Any:

        arg_0: t.Tensor = args[0].clone()

        if self.patch_mode and self.is_dest:
            assert self.patch_src_outs is not None and self.curr_src_outs is not None
            if self.mask_fn == "hard_concrete":
                mask = sample_hard_concrete(
                    self.patch_mask, arg_0.size(0), self.batch_size is not None
                )
            elif self.mask_fn == "sigmoid":
                mask = t.sigmoid(self.patch_mask)
            else:
                assert self.mask_fn is None
                mask = self.patch_mask
            mask = self.dropout_layer(mask)

            ein_pre_A, ein_pre_B, ein_post = self._get_ein_strs()

            # d = self.patch_src_outs[self.in_srcs] - self.curr_src_outs[self.in_srcs]
            # arg_0 += einsum(
            #     mask, d, f"{ein_pre_A}, {ein_pre_B} -> {ein_post}"
            # )  # Add mask times diff

            arg_0 = PatchFunction.apply(
                arg_0,
                mask,
                self.patch_src_outs,
                self.curr_src_outs,
                self.in_srcs,
                ein_pre_A,
                ein_pre_B,
                ein_post,
            )  # type: ignore

        new_args = (arg_0,) + args[1:]
        out = self.module(*new_args, **kwargs)

        if self.patch_mode and self.is_src:
            assert self.curr_src_outs is not None
            if self.head_dim is None:
                src_out = out
            else:
                squeeze_dim = self.head_dim if self.head_dim < 0 else self.head_dim + 1
                src_out = t.stack(out.split(1, dim=self.head_dim)).squeeze(squeeze_dim)
            if self.curr_src_outs.is_sparse:
                self.curr_src_outs = assign_sparse_tensor(
                    self.curr_src_outs, self.src_idxs, src_out
                )
            else:
                self.curr_src_outs[self.src_idxs] = src_out

        return out

    def _get_ein_strs(self):
        if self.mask_fn == "hard_concrete":
            batch_str = "batch"  # Sample distribution for each batch element
        elif self.mask_fn == "sigmoid":
            batch_str = ""
        else:
            batch_str = "batch" if self.batch_size is not None else ""

        head_str = "" if self.head_dim is None else "dest"  # Patch heads separately
        seq_str = "" if self.seq_dim is None else "seq"  # Patch tokens separately
        ein_pre_A = f"{batch_str} {seq_str} {head_str} src"
        ein_pre_B = f"src batch {self.dims} ..."
        ein_post = f"batch {self.dims} {head_str} ..."
        return ein_pre_A, ein_pre_B, ein_post

    def __repr__(self):
        module_str = (
            self.module.name if hasattr(self.module, "name") else self.module_name
        )
        repr = [f"PatchWrapper({module_str})"]
        repr.append(("Src✓" if self.is_src else "") + ("Dest✓" if self.is_dest else ""))
        repr.append(f"Patch Mask: [{self.patch_mask.shape}]") if self.is_dest else None
        # repr.append(str(self.patch_mask.data)) if self.is_dest else None
        return "\n".join(repr)

    def trace_handler(self, prof: t.profiler.profile):
        # Prefix for file names.
        module_str = self.module.name if hasattr(self.module, "name") else self.module
        timestamp = datetime.now().strftime(TIME_FORMAT_STR)
        file_prefix = f"{module_str}_{timestamp}"

        # Construct the trace file.
        prof.export_chrome_trace(f"profiler/{file_prefix}.json")

        # Construct the memory timeline file.
        prof.export_memory_timeline(f"profiler/{file_prefix}.html")  # device="mps"


def _calculate_diff(patch_src_outs: t.Tensor, curr_src_outs: t.Tensor, in_srcs: slice):
    if patch_src_outs.is_sparse and curr_src_outs.is_sparse:
        in_srcs_tensor = t.arange(
            in_srcs.start or 0,
            in_srcs.stop or patch_src_outs.shape[0],
            in_srcs.step or 1,
            device=patch_src_outs.device,
        )
        return (
            patch_src_outs.index_select(0, in_srcs_tensor)
            - curr_src_outs.index_select(0, in_srcs_tensor)
        ).to_dense()
    else:
        return patch_src_outs[in_srcs] - curr_src_outs[in_srcs]


class PatchFunction(t.autograd.Function):

    @staticmethod
    def forward(
        x: t.Tensor,
        mask: t.Tensor,
        patch_src_outs: t.Tensor,
        curr_src_outs: t.Tensor,
        in_srcs: slice,
        ein_pre_A: str,
        ein_pre_B: str,
        ein_post: str,
    ):
        d = _calculate_diff(patch_src_outs, curr_src_outs, in_srcs)
        return x + einsum(mask, d, f"{ein_pre_A}, {ein_pre_B} -> {ein_post}")

    @staticmethod
    def setup_context(
        ctx: t.autograd.function.FunctionCtx,
        inputs: Tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, slice, str, str, str],
        output: t.Tensor,
    ):
        (
            x,
            mask,
            patch_src_outs,
            curr_src_outs,
            in_srcs,
            ein_pre_A,
            ein_pre_B,
            ein_post,
        ) = inputs
        ctx.in_srcs = in_srcs
        ctx.ein_pre_A = ein_pre_A
        ctx.ein_pre_B = ein_pre_B
        ctx.ein_post = ein_post
        ctx.save_for_backward(patch_src_outs, curr_src_outs)
        # ctx.patch_src_outs = patch_src_outs
        # ctx.curr_src_outs = curr_src_outs

    @staticmethod
    @t.autograd.function.once_differentiable
    def backward(ctx: t.autograd.function.FunctionCtx, grad_output: t.Tensor):
        patch_src_outs, curr_src_outs = ctx.saved_tensors
        # patch_src_outs, curr_src_outs = ctx.patch_src_outs, ctx.curr_src_outs
        d = _calculate_diff(patch_src_outs, curr_src_outs, ctx.in_srcs)

        grad_x = grad_output
        grad_mask = einsum(
            grad_output, d, f"{ctx.ein_post}, {ctx.ein_pre_B} -> {ctx.ein_pre_A}"
        )

        # del ctx.patch_src_outs, ctx.curr_src_outs
        return grad_x, grad_mask, None, None, None, None, None, None


import torch as t
from torch.autograd import gradcheck


def test_patch_function():
    # Set up input tensors
    batch, dest, src, d1 = 3, 4, 2, 5
    x = t.randn(batch, d1, dest, requires_grad=True, dtype=t.float64)
    mask = t.randn(batch, dest, src, requires_grad=True, dtype=t.double)
    patch_src_outs = t.randn(src, batch, d1, dtype=t.float64)
    curr_src_outs = t.randn(src, batch, d1, dtype=t.float64)
    in_srcs = slice(0, src)
    ein_pre_A = "batch dest src"
    ein_pre_B = "src batch d1 ..."
    ein_post = "batch d1 dest ..."

    # Run gradcheck
    assert gradcheck(
        PatchFunction.apply,
        (
            x,
            mask,
            patch_src_outs,
            curr_src_outs,
            in_srcs,
            ein_pre_A,
            ein_pre_B,
            ein_post,
        ),  # type: ignore
        eps=1e-6,
        atol=1e-4,
    )

    print("PatchFunction gradcheck passed!")


if __name__ == "__main__":
    test_patch_function()
