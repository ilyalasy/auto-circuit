from typing import Any, Dict, List, Optional, Set, Tuple

import torch as t
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

from auto_circuit.types import DestNode, Edge, Node, SrcNode
from auto_circuit.utils.patch_wrapper import PatchWrapper


class PatchableModel(t.nn.Module):
    nodes: Set[Node]
    srcs: Set[SrcNode]
    dests: Set[DestNode]
    edge_dict: Dict[int | None, List[Edge]]  # Key is token position or None for all
    edges: Set[Edge]
    seq_dim: int
    seq_len: Optional[int]
    wrappers: Set[PatchWrapper]
    src_wrappers: Set[PatchWrapper]
    dest_wrappers: Set[PatchWrapper]
    out_slice: Tuple[slice | int, ...]
    is_transformer: bool
    kv_caches: Optional[Dict[int, HookedTransformerKeyValueCache]]
    wrapped_model: t.nn.Module

    def __init__(
        self,
        nodes: Set[Node],
        srcs: Set[SrcNode],
        dests: Set[DestNode],
        edge_dict: Dict[int | None, List[Edge]],
        edges: Set[Edge],
        seq_dim: int,
        seq_len: Optional[int],
        wrappers: Set[PatchWrapper],
        src_wrappers: Set[PatchWrapper],
        dest_wrappers: Set[PatchWrapper],
        out_slice: Tuple[slice | int, ...],
        is_transformer: bool,
        kv_caches: Tuple[Optional[HookedTransformerKeyValueCache], ...],
        wrapped_model: t.nn.Module,
    ) -> None:
        super().__init__()
        self.nodes = nodes
        self.srcs = srcs
        self.dests = dests
        self.edge_dict = edge_dict
        self.edges = edges
        self.seq_dim = seq_dim
        self.seq_len = seq_len
        self.wrappers = wrappers
        self.src_wrappers = src_wrappers
        self.dest_wrappers = dest_wrappers
        self.out_slice = out_slice
        self.is_transformer = is_transformer
        if all([kv_cache is None for kv_cache in kv_caches]) or len(kv_caches) == 0:
            self.kv_caches = None
        else:
            self.kv_caches = {}
            for kv_cache in kv_caches:
                if kv_cache is not None:
                    batch_size = kv_cache.previous_attention_mask.shape[0]
                    self.kv_caches[batch_size] = kv_cache
        self.wrapped_model = wrapped_model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.kv_caches is None:
            return self.wrapped_model(*args, **kwargs)
        else:
            batch_size = args[0].shape[0]
            kv = self.kv_caches[batch_size]
            return self.wrapped_model(*args, past_kv_cache=kv, **kwargs)

    def run_with_cache(self, *args: Any, **kwargs: Any) -> Any:
        if self.kv_caches is None:
            return self.wrapped_model.run_with_cache(*args, **kwargs)
        else:
            batch_size = args[0].shape[0]
            kv = self.kv_caches[batch_size]
            return self.wrapped_model.run_with_cache(*args, past_kv_cache=kv, **kwargs)

    def add_hook(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped_model.add_hook(*args, **kwargs)

    @property
    def cfg(self) -> Any:
        return self.wrapped_model.cfg

    @property
    def tokenizer(self) -> Any:
        return self.wrapped_model.tokenizer

    @property
    def input_to_embed(self) -> Any:
        return self.wrapped_model.input_to_embed

    @property
    def blocks(self) -> Any:
        return self.wrapped_model.blocks

    def to_tokens(self) -> Any:
        return self.wrapped_model.to_tokens

    def to_str_tokens(self) -> Any:
        return self.wrapped_model.to_str_tokens

    def to_string(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped_model.to_string(*args, **kwargs)

    def __str__(self) -> str:
        return self.wrapped_model.__str__()

    def __repr__(self) -> str:
        return self.wrapped_model.__repr__()
