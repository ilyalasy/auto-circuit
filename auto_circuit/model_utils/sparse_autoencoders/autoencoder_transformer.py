"""
A transformer model that patches in sparse autoencoder reconstructions at each layer.
Work in progress. Error nodes not implemented.
"""

from collections import defaultdict
from itertools import chain, count, product
from typing import Any, Dict, List, Optional, Set

import torch as t
from sae_lens import HookedSAETransformer
from tqdm import tqdm

from auto_circuit.data import PromptDataLoader
from auto_circuit.model_utils.sparse_autoencoders.sparse_autoencoder import SAEWrapper
from auto_circuit.types import AblationType, DestNode, Edge, Node, SrcNode
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import make_model_patchable
from auto_circuit.utils.patchable_model import PatchableModel


def prune_latents_with_dataset(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    max_latents: Optional[int],
    include_corrupt: bool = False,
    seq_len: Optional[int] = None,
):
    """
    Prune the weights of the autoencoder to remove latents that are never activated
    by the dataset. This can reduce the number of edges in the factorized model by a
    factor of 10 or more.
    """
    assert isinstance(
        model.wrapped_model, HookedSAETransformer
    ), "Model must be a HookedSAETransformer to prune latents with dataset."

    # src_outs = src_ablations(
    #     model,
    #     dataloader.dataset[0].clean,
    #     AblationType.RESAMPLE,
    # )
    with t.no_grad():
        src_outs = src_ablations(
            model,
            dataloader,
            (
                AblationType.TOKENWISE_MEAN_CLEAN_AND_CORRUPT
                if include_corrupt
                else AblationType.TOKENWISE_MEAN_CLEAN
            ),
            use_sparse=True,
        )

    src_modules: Dict[t.nn.Module, List[SrcNode]] = defaultdict(list)
    [src_modules[src.module(model)].append(src) for src in model.srcs]
    idx_count = count()
    new_srcs = set()
    for module, srcs in (pbar := tqdm(src_modules.items())):
        pbar.set_description_str(f"Processing Module {module}")
        for src in srcs:
            if src_outs[src.src_idx]._nnz() != 0:
                new_srcs.add(
                    SrcNode(
                        name=src.name,
                        module_name=src.module_name,
                        layer=src.layer,
                        src_idx=next(idx_count),
                        head_dim=src.head_dim,
                        head_idx=src.head_idx,
                        weight=src.weight,
                        weight_head_dim=src.weight_head_dim,
                    )
                )

    srcs = set(new_srcs)
    dests = set(model.dests)

    edge_dict: Dict[Optional[int], List[Edge]] = defaultdict(list)
    for i in [None] if seq_len is None else range(seq_len):
        pairs = product(srcs, dests)
        edge_dict[i] = [Edge(s, d, i) for s, d in pairs if s.layer < d.layer]
    nodes: Set[Node] = set(srcs | dests)
    edges = set(list(chain.from_iterable(edge_dict.values())))

    model.reset_modules()
    wrappers, src_wrappers, dest_wrappers = make_model_patchable(
        model.wrapped_model,
        model.is_factorized,
        srcs,
        nodes,
        src_outs.device,
        model.seq_len,
        model.seq_dim,
    )
    return PatchableModel(
        nodes=nodes,
        srcs=srcs,
        dests=dests,
        edge_dict=edge_dict,
        edges=edges,
        seq_dim=model.seq_dim,
        seq_len=model.seq_len,
        wrappers=wrappers,
        src_wrappers=src_wrappers,
        dest_wrappers=dest_wrappers,
        out_slice=model.out_slice,
        is_factorized=model.is_factorized,
        is_transformer=model.is_transformer,
        separate_qkv=model.separate_qkv,
        kv_caches=model.kv_caches if model.kv_caches is not None else [],
        wrapped_model=model.wrapped_model,
        ignore_tokens=model.ignore_tokens,
    )


# class AutoencoderTransformer(t.nn.Module):
#     wrapped_model: t.nn.Module
#     sparse_autoencoders: List[SparseAutoencoder]

#     def __init__(self, wrapped_model: t.nn.Module, saes: List[SparseAutoencoder]):
#         super().__init__()
#         self.sparse_autoencoders = saes

#         if isinstance(wrapped_model, PatchableModel):
#             self.wrapped_model = wrapped_model.wrapped_model
#         else:
#             self.wrapped_model = wrapped_model

#     def forward(self, *args: Any, **kwargs: Any) -> Any:
#         with t.profiler.profile(
#             profile_memory=True,
#             record_shapes=True,
#             with_stack=True,
#             on_trace_ready=t.profiler.tensorboard_trace_handler(
#                 f"profiler/run-simple-{time.time_ns()}"
#             ),
#         ) as prof:
#             return self.wrapped_model(*args, **kwargs)

#     def reset_activated_latents(
#         self, batch_len: Optional[int] = None, seq_len: Optional[int] = None
#     ):
#         for sae in self.sparse_autoencoders:
#             sae.reset_activated_latents(batch_len=batch_len, seq_len=seq_len)

#     def _prune_latents_with_dataset(
#         self,
#         dataloader: PromptDataLoader,
#         max_latents: Optional[int],
#         include_corrupt: bool = False,
#         seq_len: Optional[int] = None,
#     ):
#         """
#         !In place operation!
#         Prune the weights of the autoencoder to remove latents that are never activated
#         by the dataset. This can reduce the number of edges in the factorized model by a
#         factor of 10 or more.
#         """
#         self.reset_activated_latents(seq_len=seq_len)

#         print("Running dataset for autoencoder pruning...")
#         unpruned_logits = []
#         for batch_idx, batch in (batch_pbar := tqdm(enumerate(dataloader))):
#             batch_pbar.set_description_str(f"Pruning Autoencoder: Batch {batch_idx}")
#             for input_idx, prompt in (input_pbar := tqdm(enumerate(batch.clean))):
#                 input_pbar.set_description_str(f"Clean Batch Input {input_idx}")
#                 with t.inference_mode():
#                     out = self.forward(prompt.unsqueeze(0))  # Run one at a time
#                 unpruned_logits.append(out)
#             if include_corrupt:
#                 for input_idx, prompt in (input_pbar := tqdm(enumerate(batch.corrupt))):
#                     input_pbar.set_description_str(f"Corrupt Batch Input {input_idx}")
#                     with t.inference_mode():
#                         out = self.forward(prompt.unsqueeze(0))
#                     unpruned_logits.append(out)

#         activated_latent_counts, latent_counts = [], []
#         for sae in self.sparse_autoencoders:
#             activated = (sae.latent_total_act > 0).sum(dim=-1).tolist()
#             activated_latent_counts.append(activated)
#             activated_count = activated if type(activated) == int else max(activated)
#             max_latents = max_latents or activated_count
#             latent_counts.append(max_idx := min(max_latents, activated_count))
#             sorted_latents = t.sort(sae.latent_total_act, dim=-1, descending=True)
#             idxs_to_keep = sorted_latents.indices[..., :max_idx]
#             sae.prune_latents(idxs_to_keep)

#         pruned_logits = []
#         with t.inference_mode():
#             for batch_idx, batch in (batch_pbar := tqdm(enumerate(dataloader))):
#                 batch_pbar_str = f"Testing Pruned Autoencoder: Batch {batch_idx}"
#                 batch_pbar.set_description_str(batch_pbar_str)
#                 out = self.forward(batch.clean)
#                 pruned_logits.append(out)
#                 if include_corrupt:
#                     out = self.forward(batch.corrupt)
#                     pruned_logits.append(out)

#         flat_pruned_logits = t.flatten(t.stack(pruned_logits), end_dim=-2)
#         flat_unpruned_logits = t.flatten(t.stack(unpruned_logits), end_dim=-2)
#         kl_div = t.nn.functional.kl_div(
#             t.nn.functional.log_softmax(flat_pruned_logits, dim=-1),
#             t.nn.functional.log_softmax(flat_unpruned_logits, dim=-1),
#             reduction="batchmean",
#             log_target=True,
#         )

#         print("Done. Autoencoder activated latent counts:", activated_latent_counts)
#         print("Autoencoder latent counts:", latent_counts)
#         print("Pruned vs. Unpruned KL Div:", kl_div.item())

#     def run_with_cache(self, *args: Any, **kwargs: Any) -> Any:
#         return self.wrapped_model.run_with_cache(*args, **kwargs)

#     def run_with_hooks(self, *args: Any, **kwargs: Any) -> Any:
#         return self.wrapped_model.run_with_hooks(*args, **kwargs)

#     def add_hook(self, *args: Any, **kwargs: Any) -> Any:
#         return self.wrapped_model.add_hook(*args, **kwargs)

#     def reset_hooks(self) -> None:
#         return self.wrapped_model.reset_hooks()

#     @property
#     def cfg(self) -> Any:
#         return self.wrapped_model.cfg

#     @property
#     def tokenizer(self) -> Any:
#         return self.wrapped_model.tokenizer

#     @property
#     def input_to_embed(self) -> Any:
#         return self.wrapped_model.input_to_embed

#     @property
#     def blocks(self) -> Any:
#         return self.wrapped_model.blocks

#     def to_tokens(self, *args: Any, **kwargs: Any) -> Any:
#         return self.wrapped_model.to_tokens(*args, **kwargs)

#     def to_str_tokens(self, *args: Any, **kwargs: Any) -> Any:
#         return self.wrapped_model.to_str_tokens(*args, **kwargs)

#     def to_string(self, *args: Any, **kwargs: Any) -> Any:
#         return self.wrapped_model.to_string(*args, **kwargs)

#     def __str__(self) -> str:
#         return self.wrapped_model.__str__()

#     def __repr__(self) -> str:
#         return self.wrapped_model.__repr__()


def sae_model(
    model_name: str,
    sae_release_name: str,
    sae_id_template: str,
    device: str,
) -> HookedSAETransformer:
    """
    Load
    [`SparseAutoencoder`][auto_circuit.model_utils.sparse_autoencoders.sparse_autoencoder.SparseAutoencoder]
    wrappers into a transformer model.
    Release names and ID templates can be found in
    [list of pretrained SAEs](https://github.com/jbloomAus/SAELens/blob/main/sae_lens/pretrained_saes.yaml).
    """

    model: HookedSAETransformer = HookedSAETransformer.from_pretrained(
        model_name,
        device=device,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
    )  # type: ignore
    model.cfg.use_attn_result = True
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    for layer_idx in range(model.cfg.n_layers):
        sae, cfg_dict, log_sparsities = SAEWrapper.from_pretrained(
            sae_release_name,
            sae_id_template.format(layer_idx),
            device=device,
        )

        model.add_sae(sae)

    return model


def factorized_src_nodes(model: HookedSAETransformer) -> Set[SrcNode]:
    """Get the source part of each edge in the factorized graph, grouped by layer.
    Graph is factorized following the Mathematical Framework paper."""
    assert model.cfg.use_attn_result  # Get attention head outputs separately
    assert model.cfg.use_attn_in  # Get attention head inputs separately
    assert model.cfg.use_split_qkv_input  # Separate Q, K, V input for each head
    if not model.cfg.attn_only:
        assert model.cfg.use_hook_mlp_in  # Get MLP input BEFORE layernorm
    assert not model.cfg.attn_only

    layers, idxs = count(), count()
    nodes = set()
    nodes.add(
        SrcNode(
            name="Resid Start",
            module_name="blocks.0.hook_resid_pre",
            layer=next(layers),
            src_idx=next(idxs),
            weight="embed.W_E",
        )
    )

    sae_blocks = list(model.acts_to_saes.values())
    for block_idx in range(model.cfg.n_layers):
        layer = next(layers)
        for head_idx in range(model.cfg.n_heads):
            nodes.add(
                SrcNode(
                    name=f"A{block_idx}.{head_idx}",
                    module_name=f"blocks.{block_idx}.attn.hook_result",
                    layer=layer,
                    src_idx=next(idxs),
                    head_dim=2,
                    head_idx=head_idx,
                    weight=f"blocks.{block_idx}.attn.W_O",
                    weight_head_dim=0,
                )
            )
        layer = layer if model.cfg.parallel_attn_mlp else next(layers)
        n_latents = sae_blocks[block_idx].cfg.d_sae
        for latent_idx in range(n_latents):
            nodes.add(
                SrcNode(
                    name=f"MLP {block_idx} Latent {latent_idx}",
                    module_name=f"{sae_blocks[block_idx].name}.hook_sae_latents",
                    layer=layer,
                    src_idx=next(idxs),
                    head_dim=2,
                    head_idx=latent_idx,
                    weight=f"blocks.{block_idx}.hook_mlp_out.W_dec",
                    weight_head_dim=0,
                )
            )
    return nodes


def factorized_dest_nodes(
    model: HookedSAETransformer, separate_qkv: bool
) -> Set[DestNode]:
    """Get the destination part of each edge in the factorized graph, grouped by layer.
    Graph is factorized following the Mathematical Framework paper."""
    if separate_qkv:
        assert model.cfg.use_split_qkv_input  # Separate Q, K, V input for each head
    else:
        assert model.cfg.use_attn_in
    if not model.cfg.attn_only:
        assert model.cfg.use_hook_mlp_in  # Get MLP input BEFORE layernorm
    layers = count(1)
    nodes = set()
    for block_idx in range(model.cfg.n_layers):
        layer = next(layers)
        for head_idx in range(model.cfg.n_heads):
            if separate_qkv:
                for letter in ["Q", "K", "V"]:
                    nodes.add(
                        DestNode(
                            name=f"A{block_idx}.{head_idx}.{letter}",
                            module_name=f"blocks.{block_idx}.hook_{letter.lower()}_input",
                            layer=layer,
                            head_dim=2,
                            head_idx=head_idx,
                            weight=f"blocks.{block_idx}.attn.W_{letter}",
                            weight_head_dim=0,
                        )
                    )
            else:
                nodes.add(
                    DestNode(
                        name=f"A{block_idx}.{head_idx}",
                        module_name=f"blocks.{block_idx}.hook_attn_in",
                        layer=layer,
                        head_dim=2,
                        head_idx=head_idx,
                        weight=f"blocks.{block_idx}.attn.W_QKV",
                        weight_head_dim=0,
                    )
                )
        nodes.add(
            DestNode(
                name=f"MLP {block_idx}",
                module_name=f"blocks.{block_idx}.hook_mlp_in",
                layer=layer if model.cfg.parallel_attn_mlp else next(layers),
                weight=f"blocks.{block_idx}.mlp.W_in",
            )
        )
    nodes.add(
        DestNode(
            name="Resid End",
            module_name=f"blocks.{model.cfg.n_layers - 1}.hook_resid_post",
            layer=next(layers),
            weight="unembed.W_U",
        )
    )
    return nodes
