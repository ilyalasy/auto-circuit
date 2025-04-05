import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch as t
import torch.utils.data
from attr import dataclass
from torch.utils.data import DataLoader, Dataset, Subset
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.utils import get_attention_mask
BatchKey = int
"""A unique key for a [`PromptPairBatch`][auto_circuit.data.PromptPairBatch]."""


@dataclass(frozen=True)
class PromptPair:
    """
    A pair of clean and corrupt prompts with correct and incorrect answers.

    Args:
        clean: The 'clean' prompt, containing 'input_ids' and 'attention_mask'.
        corrupt: The 'corrupt' prompt, containing 'input_ids' and 'attention_mask'.
        answers: The correct completions for the clean prompt.
        wrong_answers: The incorrect completions for the clean prompt.
    """

    clean: Dict[str, t.Tensor]
    corrupt: Dict[str, t.Tensor]
    answers: t.Tensor
    wrong_answers: t.Tensor


@dataclass(frozen=True)
class PromptPairBatch:
    """
    A batch of prompt pairs.

    Args:
        key: A unique integer that identifies the batch.
        batch_diverge_idx: The minimum index over all prompts at which the clean and
            corrupt prompts diverge. This is used to automatically cache the key-value
            activations for the common prefix of the prompts. See
            [`load_datasets_from_json`][auto_circuit.data.load_datasets_from_json] for
            more information.
        clean: The 'clean' prompts in a dictionary with 'input_ids' and 'attention_mask'.
        corrupt: The 'corrupt' prompts in a dictionary with 'input_ids' and 'attention_mask'.
        answers: The correct answers completions for the clean prompts.
            If all prompts have the same number of answers, this is a 2D tensor.
            If each prompt has a different number of answers, this is a list of 1D
            tensors. This can make some methods such as
            [`batch_answer_diffs`][auto_circuit.utils.tensor_ops.batch_answer_diffs]
            much slower.
        wrong_answers: The incorrect answers. If each prompt has a different number of
            wrong answers, this is a list of tensors.
            If all prompts have the same number of wrong answers, this is a 2D tensor.
            If each prompt has a different number of wrong answers, this is a list of 1D
            tensors. This can make some methods such as
            [`batch_answer_diffs`][auto_circuit.utils.tensor_ops.batch_answer_diffs]
            much slower.
    """

    key: BatchKey
    batch_diverge_idx: int
    clean: Dict[str, t.Tensor]
    corrupt: Dict[str, t.Tensor]
    answers: List[t.Tensor] | t.Tensor
    wrong_answers: List[t.Tensor] | t.Tensor


def collate_fn(batch: List[PromptPair]) -> PromptPairBatch:
    # Stack input_ids and attention_masks separately
    clean_input_ids = t.stack([p.clean["input_ids"] for p in batch])
    corrupt_input_ids = t.stack([p.corrupt["input_ids"] for p in batch])
    
    clean_attention_mask = None
    corrupt_attention_mask = None
    
    if all("attention_mask" in p.clean for p in batch):
        clean_attention_mask = t.stack([p.clean["attention_mask"] for p in batch])
    
    if all("attention_mask" in p.corrupt for p in batch):
        corrupt_attention_mask = t.stack([p.corrupt["attention_mask"] for p in batch])
    
    # Handle answers and wrong_answers
    if all([p.answers.shape == batch[0].answers.shape for p in batch]):
        answers = t.stack([p.answers for p in batch])
    else:  # Sometimes each prompt has a different number of answers
        answers = [p.answers for p in batch]
    
    if all([p.wrong_answers.shape == batch[0].wrong_answers.shape for p in batch]):
        wrong_answers = t.stack([p.wrong_answers for p in batch])
    else:  # Sometimes each prompt has a different number of wrong answers
        wrong_answers = [p.wrong_answers for p in batch]

    # Create clean and corrupt dictionaries
    clean = {"input": clean_input_ids} # TransformerLens argument is "input", instead of "input_ids"
    corrupt = {"input": corrupt_input_ids}
    
    if clean_attention_mask is not None:
        clean["attention_mask"] = clean_attention_mask
        
    if corrupt_attention_mask is not None:
        corrupt["attention_mask"] = corrupt_attention_mask

    # Generate hash key based on input_ids
    key = hash((str(clean_input_ids.tolist()), str(corrupt_input_ids.tolist())))

    # Find diverge index
    diverge_idxs = (~(clean_input_ids == corrupt_input_ids)).int().argmax(dim=1)
    batch_dvrg_idx: int = int(diverge_idxs.min().item())
    
    return PromptPairBatch(key, batch_dvrg_idx, clean, corrupt, answers, wrong_answers)


class PromptDataset(Dataset):
    def __init__(
        self,
        clean_prompts: List[Dict[str, t.Tensor]] | Dict[str, t.Tensor],
        corrupt_prompts: List[Dict[str, t.Tensor]] | Dict[str, t.Tensor],
        answers: List[t.Tensor],
        wrong_answers: List[t.Tensor],
    ):
        """
        A dataset of clean/corrupt prompt pairs with correct/incorrect answers.

        Args:
            clean_prompts: The 'clean' prompts with 'input_ids' and 'attention_mask'.
                If a list, each element is a dictionary with tensors.
                If a dictionary, keys are 'input_ids' and 'attention_mask' with tensors
                of shape (n_prompts, prompt_length).
            corrupt_prompts: The 'corrupt' prompts with 'input_ids' and 'attention_mask'.
                If a list, each element is a dictionary with tensors.
                If a dictionary, keys are 'input_ids' and 'attention_mask' with tensors
                of shape (n_prompts, prompt_length).
            answers: A list of correct answers.
                Each element is a 1D tensor with the answer tokens.
            wrong_answers: A list of incorrect answers.
                Each element is a 1D tensor with the wrong answer tokens.
        """
        # Convert dictionary form to list form if necessary
        if isinstance(clean_prompts, dict):
            n_prompts = len(clean_prompts["input_ids"])
            self.clean_prompts = [
                {k: v[i] for k, v in clean_prompts.items()}
                for i in range(n_prompts)
            ]
        else:
            self.clean_prompts = clean_prompts
            
        if isinstance(corrupt_prompts, dict):
            n_prompts = len(corrupt_prompts["input_ids"])
            self.corrupt_prompts = [
                {k: v[i] for k, v in corrupt_prompts.items()}
                for i in range(n_prompts)
            ]
        else:
            self.corrupt_prompts = corrupt_prompts
            
        self.answers = answers
        self.wrong_answers = wrong_answers

    def __len__(self) -> int:
        return len(self.clean_prompts)

    def __getitem__(self, idx: int) -> PromptPair:
        return PromptPair(
            self.clean_prompts[idx],
            self.corrupt_prompts[idx],
            self.answers[idx],
            self.wrong_answers[idx],
        )


class PromptDataLoader(DataLoader[PromptPairBatch]):
    def __init__(
        self,
        prompt_dataset: Any,
        seq_len: Optional[int],
        diverge_idx: int,
        kv_cache: Optional[HookedTransformerKeyValueCache] = None,
        seq_labels: Optional[List[str]] = None,
        word_idxs: Dict[str, int] = {},
        **kwargs: Any,
    ):
        """
        A `DataLoader` for clean/corrupt prompt pairs with correct/incorrect answers.

        Args:
            prompt_dataset: A [`PromptDataset`][auto_circuit.data.PromptDataset] with
                clean and corrupt prompts.
            seq_len: The token length of the prompts (if fixed length). This prompt
                length can be passed to `patchable_model` to enable patching specific
                token positions.
            diverge_idx: The index at which the clean and corrupt prompts diverge. (See
                [`load_datasets_from_json`][auto_circuit.data.load_datasets_from_json]
                for more information.)
            kv_cache: A cache of past key-value pairs for the transformer. Only used if
                `diverge_idx` is greater than 0. (See
                [`load_datasets_from_json`][auto_circuit.data.load_datasets_from_json]
                for more information.)
            seq_labels: A list of strings that label each token for fixed length
                prompts. Used by
                [`draw_seq_graph`][auto_circuit.visualize.draw_seq_graph] to label the
                circuit diagram.
            word_idxs: A dictionary with the token indexes of specific words. Used by
                official circuit functions.
            kwargs: Additional arguments to pass to `DataLoader`.

        Note:
            `drop_last=True` is always passed to the parent `DataLoader` constructor. So
            all batches are always the same size. This simplifies the implementation of
            several functions. For example, the `kv_cache` only needs caches for a
            single batch size.
        """
        super().__init__(
            prompt_dataset, **kwargs, drop_last=True, collate_fn=collate_fn
        )
        self.seq_len = seq_len
        """
        The token length of the prompts (if fixed length). This prompt length can be
        passed to `patchable_model` to enable patching specific token positions.
        """
        self.diverge_idx = diverge_idx
        """
        The index at which the clean and corrupt prompts diverge. (See
        [`load_datasets_from_json`][auto_circuit.data.load_datasets_from_json] for more
        information.)
        """
        self.seq_labels = seq_labels
        """
        A list of strings that label each token for fixed length prompts. Used by
        [`draw_seq_graph`][auto_circuit.visualize.draw_seq_graph] to label the circuit
        diagram.
        """
        assert kv_cache is None or diverge_idx > 0
        self.kv_cache = kv_cache
        """
        A cache of past key-value pairs for the transformer. Only used if `diverge_idx`
        is greater than 0. (See
        [`load_datasets_from_json`][auto_circuit.data.load_datasets_from_json] for more
        information.)
        """
        self.word_idxs = word_idxs
        """
        A dictionary with the token indexes of specific words. Used by official circuit
        functions.
        """


def load_datasets_from_json(
    model: Optional[HookedTransformer],
    path: Path | List[Path],
    device: t.device,
    prepend_bos: bool = True,
    batch_size: int | Tuple[int, int] = 32,  # (train, test) if tuple
    train_test_size: Tuple[int, int] = (128, 128),
    return_seq_length: bool = False,
    tail_divergence: bool = False,  # Remove all tokens before divergence
    shuffle: bool = True,
    random_seed: int = 42,
    pad: bool = True,
) -> Tuple[PromptDataLoader, PromptDataLoader]:
    """
    Load a dataset from a json file. The file should specify a list of
    dictionaries with keys "clean_prompt" and "corrupt_prompt".

    JSON data format:
    ```
    {
        // Optional: used to label circuit visualization
        "seq_labels": [str, ...],

        // Optional: used by official circuit functions
        "word_idxs": {
            str: int,
            ...
        },

        // Required: the prompt pairs
        "prompts": [
            {
                "clean": str | [[int, ...], ...],
                "corrupt": str | [[int, ...], ...],
                "answers": [str, ...] | [int, ...],
                "wrong_answers": [str, ...] | [int, ...],
            },
            ...
        ]
    }
    ```

    Args:
        model: Model to use for tokenization. If None, data must be pre-tokenized
            (`"prompts"` is passed as `int`s).
        path: Path to the json file with the dataset. If a list of paths is passed, the
            first dataset is parsed in full and for the rest are the `prompts` are used.
        device: Device to load the data on.
        prepend_bos: If True, prepend the `BOS` token to each prompt. (The `prepend_bos`
            flag on TransformerLens `HookedTransformer`s is ignored.)
        batch_size: The batch size for training and testing. If a single int is passed,
            the same batch size is used for both.
        return_seq_length: If `True`, return the sequence length of the prompts. **Note:
            If `True`, all the prompts must have the same length or an error will be
            raised.** This is used by
            [`patchable_model`][auto_circuit.utils.graph_utils.patchable_model] to
            enable patching specific token positions.
        tail_divergence: If all prompts share a common prefix, remove it and compute the
            keys and values for each attention head on the prefix. A `kv_cache` for the
            prefix is returned in the `train_loader` and `test_loader`.
        shuffle: If `True`, shuffle the dataset before splitting into train and test
            sets.
        random_seed: Seed for the random number generator.
        pad: If `True`, pad the prompts to the maximum length in the batch. Do not use
            in conjunction with `return_seq_length`.

    Note:
        `shuffle` only shuffles the order of the prompts once at the beginning. The
        order is preserved in the train and test loaders (`shuffle=False` is always
        passed to the [`PromptDataLoader`][auto_circuit.data.PromptDataLoader]
        constructor). This makes it easier to ensure experiments are deterministic.
    """
    assert not (prepend_bos and (model is None)), "Need model tokenizer to prepend bos"

    # Load a dataset. If path is a list, only the first dataset is fully loaded.
    first_path = path if isinstance(path, Path) else path[0]
    assert isinstance(first_path, Path)
    with open(first_path, "r") as f:
        data = json.load(f)
    # For other paths, only 'prompts' are added to dataset. (eg. seq_labels is ignored)
    if isinstance(path, list):
        assert all([isinstance(p, Path) for p in path])
        for p in path[1:]:
            with open(p, "r") as f:
                d = json.load(f)
                data["prompts"].extend(d["prompts"])

    # Shuffle data and split into train and test
    random.seed(random_seed)
    t.random.manual_seed(random_seed)
    random.shuffle(data["prompts"]) if shuffle else None
    n_train_and_test = sum(train_test_size)
    clean_prompts = [d["clean"] for d in data["prompts"]][:n_train_and_test]
    corrupt_prompts = [d["corrupt"] for d in data["prompts"]][:n_train_and_test]
    answer_strs = [d["answers"] for d in data["prompts"]][:n_train_and_test]
    wrong_answer_strs = [d["wrong_answers"] for d in data["prompts"]][:n_train_and_test]
    seq_labels = data.get("seq_labels", None)
    word_idxs = data.get("word_idxs", {})

    if prepend_bos:
        # Adjust word_idxs and seq_labels if prepending bos
        seq_labels = ["<|BOS|>"] + seq_labels if seq_labels is not None else None
        word_idxs = {k: v + int(prepend_bos) for k, v in word_idxs.items()}

    kvs = []
    diverge_idx: int = 0

    if model is None:
        # Create dictionaries for clean and corrupt prompts
        clean_prompts = [
            {"input_ids": t.tensor(p).to(device)} 
            for p in clean_prompts
        ]
        corrupt_prompts = [
            {"input_ids": t.tensor(p).to(device)} 
            for p in corrupt_prompts
        ]
        answers = [t.tensor(a).to(device) for a in answer_strs]
        wrong_answers = [t.tensor(a).to(device) for a in wrong_answer_strs]
        seq_len = clean_prompts[0]["input_ids"].shape[0]
        assert not tail_divergence
    else:
        model.tokenizer.padding_side = "left"
        clean_tokens = model.to_tokens(clean_prompts, prepend_bos=True, truncate=(model.cfg.n_ctx is not None))        
        clean_attn_mask = get_attention_mask(model.tokenizer, clean_tokens, True)
        corrupt_tokens = model.to_tokens(corrupt_prompts, prepend_bos=True, truncate=(model.cfg.n_ctx is not None))
        corrupt_attn_mask = get_attention_mask(model.tokenizer, corrupt_tokens, True)

        seq_len = None
        if return_seq_length:
            # We still want to check if the attention masks are all ones
            assert t.all(clean_attn_mask == 1)
            assert t.all(corrupt_attn_mask == 1)
            seq_len = clean_tokens.shape[1]
            
        answers = [model.to_tokens(a,prepend_bos=False).squeeze(0) for a in answer_strs]
        wrong_answers = [model.to_tokens(a,prepend_bos=False).squeeze(0) for a in wrong_answer_strs]
        
        # Create dictionaries for clean and corrupt prompts with input_ids and attention_mask
        clean_prompts = {
            "input_ids": clean_tokens,
            "attention_mask": clean_attn_mask
        }
        
        corrupt_prompts = {
            "input_ids": corrupt_tokens,
            "attention_mask": corrupt_attn_mask
        }        

        if tail_divergence:
            # Get divergence point based on input_ids
            diverge_idxs = (~(clean_prompts["input_ids"] == corrupt_prompts["input_ids"])).int().argmax(dim=1)
            diverge_idx = int(diverge_idxs.min().item())
            
        if diverge_idx > 0:
            seq_labels = seq_labels[diverge_idx:] if seq_labels is not None else None
            prefixs, cfg, device = [], model.cfg, model.cfg.device
            if isinstance(batch_size, tuple):
                prefixs.append(clean_prompts["input_ids"][: (bs0 := batch_size[0]), :diverge_idx])
                prefixs.append(clean_prompts["input_ids"][: (bs1 := batch_size[1]), :diverge_idx])
                kvs.append(HookedTransformerKeyValueCache.init_cache(cfg, device, bs0))
                kvs.append(HookedTransformerKeyValueCache.init_cache(cfg, device, bs1))
            else:
                prefixs.append(clean_prompts["input_ids"][:batch_size, :diverge_idx])
                kvs.append(
                    HookedTransformerKeyValueCache.init_cache(cfg, device, batch_size)
                )

            for i, (prefix, kv_cache) in enumerate(zip(prefixs, kvs)):
                with t.inference_mode():
                    # Pass attention mask for the prefix
                    if isinstance(batch_size, tuple):
                        bs = batch_size[i]
                    else:
                        bs = batch_size
                    prefix_attn_mask = clean_prompts["attention_mask"][:bs, :diverge_idx]
                    model(prefix, attention_mask=prefix_attn_mask, past_kv_cache=kv_cache)
                kv_cache.freeze()

            print("seq_len before divergence", seq_len)
            if return_seq_length:
                assert seq_len is not None
                seq_len -= diverge_idx
            print("seq_len after divergence", seq_len)

            # This must be done AFTER gathering the kv caches
            # For both clean and corrupt prompts, truncate the tensors to keep only after divergence
            clean_prompts = {
                "input_ids": clean_prompts["input_ids"][:, diverge_idx:],
                "attention_mask": clean_prompts["attention_mask"][:, diverge_idx:]
            }
            
            corrupt_prompts = {
                "input_ids": corrupt_prompts["input_ids"][:, diverge_idx:],
                "attention_mask": corrupt_prompts["attention_mask"][:, diverge_idx:]
            }

    dataset = PromptDataset(
        clean_prompts, corrupt_prompts, answers, wrong_answers
    )
    train_set = Subset(dataset, list(range(train_test_size[0])))
    test_set = Subset(dataset, list(range(train_test_size[0], n_train_and_test)))
    train_loader = PromptDataLoader(
        train_set,
        seq_len=seq_len,
        diverge_idx=diverge_idx,
        kv_cache=kvs[0] if len(kvs) > 0 else None,
        seq_labels=seq_labels,
        word_idxs=word_idxs,
        batch_size=batch_size[0] if isinstance(batch_size, tuple) else batch_size,
        shuffle=False,
    )
    test_loader = PromptDataLoader(
        test_set,
        seq_len=seq_len,
        diverge_idx=diverge_idx,
        kv_cache=kvs[-1] if len(kvs) > 0 else None,
        seq_labels=seq_labels,
        word_idxs=word_idxs,
        batch_size=batch_size[1] if isinstance(batch_size, tuple) else batch_size,
        shuffle=False,
    )
    return train_loader, test_loader
