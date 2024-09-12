from collections import defaultdict
from functools import partial
from typing import Dict, List, Literal, Optional

import torch as t
from transformer_lens.hook_points import HookPoint

from auto_circuit.data import BatchKey, PromptDataLoader
from auto_circuit.types import AblationType, SrcNode
from auto_circuit.utils.patchable_model import PatchableModel


class LazyAblations:
    """
    A lazy container for source nodes activations.

    This class computes and stores activations used to ablate each
    [`Edge`][auto_circuit.types.Edge] in a model, given a particular set of model inputs
    and an ablation type.
    See[`AblationType`][auto_circuit.types.AblationType] for the different types of
    ablations that can be computed.

    Will not store activations in memory if there are too many source nodes (e.g. SAE).

    Attributes:
        data (t.Tensor): Cached source node activations, if precomputed, else will be computed on access.
    """

    def __init__(
        self,
        model: PatchableModel,
        sample: t.Tensor | PromptDataLoader,
        ablation_type: AblationType,
    ):
        self._model = model
        self._sample = sample
        self._ablation_type = ablation_type
        self._data: Optional[t.Tensor] = None
        # if a lot of source nodes, we don't want to store the activations in memory
        if len(self._model.srcs) < 20000:
            self._data = self._compute()

    @property
    def data(self) -> t.Tensor:
        if self._data is None:
            return self._compute()
        return self._data

    @t.no_grad()
    def _compute(self):
        return src_ablations(self._model, self._sample, self._ablation_type)


def src_out_hook(
    out: t.Tensor,
    hook: HookPoint,
    src_nodes: List[SrcNode],
    src_outs: Dict[SrcNode, t.Tensor],
    ablation_type: AblationType,
):
    assert not ablation_type.mean_over_dataset
    if ablation_type == AblationType.RESAMPLE:
        out = out
    elif ablation_type == AblationType.ZERO:
        out = t.zeros_like(out)
    elif ablation_type == AblationType.BATCH_TOKENWISE_MEAN:
        repeats = [out.size(0)] + [1] * (out.ndim - 1)
        out = out.mean(dim=0, keepdim=True).repeat(repeats)
    elif ablation_type == AblationType.BATCH_ALL_TOK_MEAN:
        repeats = [out.size(0), out.size(1)] + [1] * (out.ndim - 2)
        out = out.mean(dim=(0, 1), keepdim=True).repeat(repeats)
    else:
        raise NotImplementedError(ablation_type)

    head_dim: Optional[int] = src_nodes[0].head_dim
    out = out if head_dim is None else out.split(1, dim=head_dim)
    for s in src_nodes:
        src_outs[s] = out if head_dim is None else out[s.head_idx].squeeze(s.head_dim)


def mean_src_out_hook(
    out: t.Tensor,
    hook: HookPoint,
    src_nodes: List[SrcNode],
    src_outs: Dict[SrcNode, t.Tensor],
    ablation_type: AblationType,
):
    assert ablation_type.mean_over_dataset
    repeats = [out.size(0)] + [1] * (out.ndim - 1)
    out = out.mean(dim=0, keepdim=True).repeat(repeats)

    head_dim: Optional[int] = src_nodes[0].head_dim
    out = out if head_dim is None else out.split(1, dim=head_dim)
    for s in src_nodes:
        src_out = out if head_dim is None else out[s.head_idx].squeeze(s.head_dim)
        if s not in src_outs:
            src_outs[s] = src_out
        else:
            src_outs[s] += src_out


def src_ablations(
    model: PatchableModel,
    sample: t.Tensor | PromptDataLoader,
    ablation_type: AblationType = AblationType.RESAMPLE,
) -> t.Tensor:
    """
    Get the activations used to ablate each [`Edge`][auto_circuit.types.Edge] in a
    model, given a particular set of model inputs and an ablation type. See
    [`AblationType`][auto_circuit.types.AblationType] for the different types of
    ablations that can be computed.

    Args:
        model: The model to get the ablations for.
        sample: The data sample to get the ablations for. This is not used for all
            `ablation_type`s. Either a single batch of inputs or a DataLoader.
        ablation_type: The type of ablation to perform.

    Returns:
        A tensor of activations used to ablated each [`Edge`][auto_circuit.types.Edge]
            model on the given input.  Shape is `[Srcs, ...]` where `Srcs` is the number
            of [`SrcNode`][auto_circuit.types.SrcNode]s in the model and `...` is the
            shape of the activations of the model. In a transformer this will be
            `[Srcs, batch, seq, d_model]`.
    """
    src_outs: Dict[SrcNode, t.Tensor] = {}
    src_modules: Dict[t.nn.Module, List[SrcNode]] = defaultdict(list)
    [src_modules[src.module(model)].append(src) for src in model.srcs]
    hooks = []
    for mod, src_nodes in src_modules.items():
        hook_fn = partial(
            mean_src_out_hook if ablation_type.mean_over_dataset else src_out_hook,
            src_nodes=src_nodes,
            src_outs=src_outs,
            ablation_type=ablation_type,
        )
        hooks.append((mod.module_name, hook_fn))

    if ablation_type.mean_over_dataset:
        # Collect activations over the entire dataset a nd take the mean
        assert isinstance(sample, PromptDataLoader)
        for batch in sample:
            if ablation_type.clean_dataset:
                model.run_with_hooks(batch.clean, fwd_hooks=hooks)
            if ablation_type.corrupt_dataset:
                model.run_with_hooks(batch.corrupt, fwd_hooks=hooks)
        # PromptDataLoader has equal size batches, so we can take the mean of means
        mult = int(ablation_type.clean_dataset) + int(ablation_type.corrupt_dataset)
        assert mult == 2 or mult == 1
        for src, src_out in src_outs.items():
            src_outs[src] = src_out / (len(sample) * mult)
    else:
        # Collect activations for a single batch
        assert isinstance(sample, t.Tensor)
        model.run_with_hooks(sample, fwd_hooks=hooks)
    # Sort the src_outs dict by node idx
    src_outs = dict(sorted(src_outs.items(), key=lambda x: x[0].src_idx))
    assert [src.src_idx for src in src_outs.keys()] == list(range(len(src_outs)))
    return t.stack(list(src_outs.values())).detach()


def src_ablations_lazy(
    model: PatchableModel,
    sample: t.Tensor | PromptDataLoader,
    ablation_type: AblationType = AblationType.RESAMPLE,
) -> LazyAblations:
    """
    Get the activations used to ablate each [`Edge`][auto_circuit.types.Edge] in a
    model, given a particular set of model inputs and an ablation type. See
    [`AblationType`][auto_circuit.types.AblationType] for the different types of
    ablations that can be computed.

    Args:
        model: The model to get the ablations for.
        sample: The data sample to get the ablations for. This is not used for all
            `ablation_type`s. Either a single batch of inputs or a DataLoader.
        ablation_type: The type of ablation to perform.

    Returns:
        A LazyAblations object containing activations used to ablate each
        [`Edge`][auto_circuit.types.Edge] in the model on the given input.
        When accessed, the data property will have shape `[Srcs, ...]` where `Srcs`
        is the number of [`SrcNode`][auto_circuit.types.SrcNode]s in the model and
        `...` is the shape of the activations of the model. In a transformer this
        will be `[Srcs, batch, seq, d_model]`.
    """

    return LazyAblations(model, sample, ablation_type)


def batch_src_ablations(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    ablation_type: AblationType = AblationType.RESAMPLE,
    clean_corrupt: Optional[Literal["clean", "corrupt"]] = None,
) -> Dict[BatchKey, LazyAblations]:
    """
    Wrapper of [`src_ablations`][auto_circuit.utils.ablation_activations.src_ablations]
    that returns ablations for each batch in a dataloader.

    Args:
        model: The model to get the ablations for.
        dataloader: The input data to get the ablations for.
        ablation_type: The type of ablation to perform.
        clean_corrupt: Whether to use the clean or corrupt inputs to calculate the
            ablations.

    Returns:
        A dictionary mapping [`BatchKey`][auto_circuit.data.BatchKey]s to the
            activations used to ablate each [`Edge`][auto_circuit.types.Edge] in the
            model on the corresponding batch.
    """
    batch_specific_ablation = [
        AblationType.RESAMPLE,
        AblationType.BATCH_TOKENWISE_MEAN,
        AblationType.BATCH_ALL_TOK_MEAN,
    ]
    assert (clean_corrupt is not None) == (ablation_type in batch_specific_ablation)

    patch_outs: Dict[BatchKey, LazyAblations] = {}
    if ablation_type.mean_over_dataset:
        mean_patch = src_ablations_lazy(model, dataloader, ablation_type)
        patch_outs = {batch.key: mean_patch for batch in dataloader}
    else:
        for batch in dataloader:
            if ablation_type == AblationType.ZERO:
                input_batch = batch.clean
            else:
                input_batch = batch.clean if clean_corrupt == "clean" else batch.corrupt
            patch_outs[batch.key] = src_ablations_lazy(
                model, input_batch, ablation_type
            )
    return patch_outs
