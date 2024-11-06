import math

import torch as t

from auto_circuit.data import PromptPairBatch
from auto_circuit.types import PruneScores

# Copied from Subnetwork Probing paper: https://github.com/stevenxcao/subnetwork-probing
left, right, temp = -0.1, 1.1, 2 / 3


def sample_hard_concrete(
    mask: t.Tensor, batch_size: int, mask_expanded: bool = False
) -> t.Tensor:
    """
    Sample from the hard concrete distribution
    ([Louizos et al., 2017](https://arxiv.org/abs/1712.01312)).

    Args:
        mask: The mask whose values parameterize the distribution.
        batch_size: The number of samples to draw.
        mask_expanded: Whether the mask has a batch dimension at the start.

    Returns:
        A sample for each element in the mask for each batch element. The returned
        tensor has shape `(batch_size, *mask.shape)`.
    """
    if not mask_expanded:
        mask = mask.repeat(batch_size, *([1] * mask.ndim))
    else:
        assert mask.size(0) == batch_size
    u = t.zeros_like(mask).uniform_().clamp(0.0001, 0.9999)
    s = t.sigmoid((u.log() - (1 - u).log() + mask) / temp)
    s_bar = s * (right - left) + left
    return s_bar.clamp(min=0.0, max=1.0)


def indices_vals(vals: t.Tensor, indices: t.Tensor) -> t.Tensor:
    assert vals.ndim == indices.ndim
    return t.gather(vals, dim=-1, index=indices)


def vocab_avg_val(vals: t.Tensor, indices: t.Tensor) -> t.Tensor:
    return indices_vals(vals, indices).mean()


def batch_avg_answer_val(
    vals: t.Tensor, batch: PromptPairBatch, wrong_answer: bool = False
) -> t.Tensor:
    """
    Get the average value of the logits (or some function of them) for the correct
    answers in the batch.

    Args:
        vals: The logits values or some tensor of the same shape.
        batch: The batch of prompts and answers.
        wrong_answer: Whether to get the average value of the wrong answers instead of
            the correct answers.

    Returns:
        The average value of the logits for the correct answers in the batch.
    """
    answers = batch.answers if not wrong_answer else batch.wrong_answers
    if isinstance(answers, t.Tensor):
        return vocab_avg_val(vals, answers)
    else:
        # If each prompt has a different number of answers we have a list of tensor
        assert isinstance(answers, list)
        return t.stack([vocab_avg_val(v, a) for v, a in zip(vals, answers)]).mean()


def batch_answer_diffs(vals: t.Tensor, batch: PromptPairBatch) -> t.Tensor:
    """
    Find the difference between the average value of the correct answers and the average
    value of the wrong answers for each prompt in the batch.

    If the batch answers are a `List`, rather than a `Tensor`, the function will be much
    slower.

    Args:
        vals: The logits values or some tensor of the same shape.
        batch: The batch of prompts and answers.

    Returns:
        The difference between the average value of the correct answers and the average
        value of the wrong answers for each prompt in the batch.
    """
    answers = batch.answers
    wrong_answers = batch.wrong_answers
    if isinstance(answers, t.Tensor) and isinstance(wrong_answers, t.Tensor):
        # We don't use vocab_avg_val here because we need to calculate the average
        # difference between the correct and wrong answers not the difference between
        # the average correct and average incorrect answers
        # We do take the mean over each set of correct and incorrect answers (often
        # there is only one of each, eg. in the IOI task).
        ans_avgs = t.gather(vals, dim=-1, index=answers).mean(dim=-1)
        wrong_avgs = t.gather(vals, dim=-1, index=wrong_answers).mean(dim=-1)
        return ans_avgs - wrong_avgs
    else:
        # If each prompt has a different number of answers we have a list of tensors
        # assert isinstance(answers, list) and isinstance(wrong_answers, list)
        ans_avgs = [vocab_avg_val(v, a) for v, a in zip(vals, answers)]
        wrong_avgs = [vocab_avg_val(v, w) for v, w in zip(vals, wrong_answers)]
        return t.stack(ans_avgs) - t.stack(wrong_avgs)


def batch_avg_answer_diff(vals: t.Tensor, batch: PromptPairBatch) -> t.Tensor:
    """
    Wrapper of [`batch_answer_diffs`][auto_circuit.utils.tensor_ops.batch_answer_diffs]
    that returns the mean of the differences.
    """
    return batch_answer_diffs(vals, batch).mean()


def batch_answer_diff_percents(
    pred_vals: t.Tensor, target_vals: t.Tensor, batch: PromptPairBatch
) -> t.Tensor:
    """
    Find the percentage difference between the predicted logit differences and the
    target logit differences.

    Args:
        pred_vals: The predicted logit values or some tensor of the same shape.
        target_vals: The target logit values or some tensor of the same shape.
        batch: The batch of prompts and answers.

    Returns:
        The percentage difference between the predicted logit differences and the target
        logit differences.
    """
    target_answer_diff = batch_answer_diffs(target_vals, batch)
    pred_answer_diff = batch_answer_diffs(pred_vals, batch)
    return (pred_answer_diff / target_answer_diff) * 100


def correct_answer_proportion(logits: t.Tensor, batch: PromptPairBatch) -> t.Tensor:
    """
    What proportion of the logits have the correct answer as the maximum?

    Args:
        logits: The logits values or some tensor of the same shape.
        batch: The batch of prompts and answers.

    Returns:
        The proportion of the logits that have the correct answer as the maximum.
    """
    answers = batch.answers
    if isinstance(answers, t.Tensor):
        assert answers.shape[-1] == 1
        max_idxs = t.argmax(logits, dim=-1, keepdim=True)
        return (max_idxs == answers).float().mean()
    else:
        # If each prompt has a different number of answers we have a list of tensors
        assert isinstance(answers, list)
        corrects = []
        for prompt_idx, prompt_answer in enumerate(answers):
            if prompt_answer.size(0) > 1:
                prompt_answer = prompt_answer[:1]  # take first token
            assert prompt_answer.shape == (1,)
            corrects.append((t.argmax(logits[prompt_idx], dim=-1) == prompt_answer))
        return t.stack(corrects).float().mean()


def correct_answer_greater_than_incorrect_proportion(
    logits: t.Tensor, batch: PromptPairBatch
) -> t.Tensor:
    """
    What proportion of the logits have the correct answer with a greater value than all
    the wrong answers?

    Args:
        logits: The logits values or some tensor of the same shape.
        batch: The batch of prompts and answers.

    Returns:
        The proportion of the logits that have the correct answer with a greater value
        than all the wrong answers.
    """
    answers = batch.answers
    wrong_answers = batch.wrong_answers
    if isinstance(answers, t.Tensor) and isinstance(wrong_answers, t.Tensor):
        assert answers.shape[-1] == 1
        answer_logits = t.gather(logits, dim=-1, index=answers)
        wrong_logits = t.gather(logits, dim=-1, index=wrong_answers)
        combined_logits = t.cat([answer_logits, wrong_logits], dim=-1)
        max_idxs = combined_logits.argmax(dim=-1)
        return (max_idxs == 0).float().mean()
    else:
        assert isinstance(answers, list) and isinstance(wrong_answers, list)
        corrects = []
        for i, (prompt_ans, prompt_wrong_ans) in enumerate(zip(answers, wrong_answers)):
            assert prompt_ans.shape == (1,)
            answer_logits = t.gather(logits[i], dim=-1, index=prompt_ans)
            wrong_logits = t.gather(logits[i], dim=-1, index=prompt_wrong_ans)
            combined_logits = t.cat([answer_logits, wrong_logits], dim=-1)
            max_idxs = combined_logits.argmax(dim=-1)
            corrects.append(max_idxs == 0)
        return t.stack(corrects).float().mean()


def multibatch_kl_div(input_logprobs: t.Tensor, target_logprobs: t.Tensor) -> t.Tensor:
    """
    Compute the average KL divergence between two sets of log probabilities.
    Assumes the last dimension of `input_logprobs` and `target_logprobs` is the log
    probability of each class. The other dimensions are batch dimensions.

    Args:
        input_logprobs: The input log probabilities.
        target_logprobs: The target log probabilities.

    Returns:
        The average KL divergence between the input and target log probabilities.
    """
    assert input_logprobs.shape == target_logprobs.shape
    kl_div_sum = t.nn.functional.kl_div(
        input_logprobs,
        target_logprobs,
        reduction="sum",
        log_target=True,
    )
    n_batch = math.prod(input_logprobs.shape[:-1])
    return kl_div_sum / n_batch


def flat_prune_scores(prune_scores: PruneScores) -> t.Tensor:
    """
    Flatten the prune scores into a single, 1-dimensional tensor.

    Args:
        prune_scores: The prune scores to flatten.

    Returns:
        The flattened prune scores.
    """
    return t.cat([ps.flatten() for _, ps in prune_scores.items()])


def desc_prune_scores(prune_scores: PruneScores) -> t.Tensor:
    """
    Flatten the prune scores into a single, 1-dimensional tensor and sort them in
    descending order.

    Args:
        prune_scores: The prune scores to flatten and sort.

    Returns:
        The flattened and sorted prune scores.
    """
    return flat_prune_scores(prune_scores).abs().sort(descending=True).values


def prune_scores_threshold(
    prune_scores: PruneScores | t.Tensor, edge_count: int
) -> t.Tensor:
    """
    Return the minimum absolute value of the top `edge_count` prune scores.
    Supports passing in a pre-sorted tensor of prune scores to avoid re-sorting.

    Args:
        prune_scores: The prune scores to threshold.
        edge_count: The number of edges that should be above the threshold.

    Returns:
        The threshold value.
    """
    if edge_count == 0:
        return t.tensor(float("inf"))  # return the maximum value so no edges are pruned

    if isinstance(prune_scores, t.Tensor):
        assert prune_scores.ndim == 1
        return prune_scores[edge_count - 1]
    else:
        return desc_prune_scores(prune_scores)[edge_count - 1]


def assign_sparse_tensor(
    sparse_tensor: t.Tensor, indices: t.Tensor | slice, values: t.Tensor
) -> t.Tensor:
    """
    Assign values to specific indices in a sparse tensor.
    """
    assert sparse_tensor.is_sparse, "Input tensor must be sparse"
    sparse_tensor = sparse_tensor.coalesce()
    # Convert slice to tensor if necessary
    if isinstance(indices, slice):
        indices = t.arange(
            indices.start or 0,
            indices.stop or sparse_tensor.shape[0],
            indices.step or 1,
        )
    # Assert that values size is the same as sparse_tensor[indices]
    # Calculate expected size of values
    expected_sizes = [
        t.Size([indices.size(0)] + list(sparse_tensor.size()[1:])),
        sparse_tensor.size()[1:] or t.Size([1]),
    ]
    # Check if values size matches expected size
    assert (
        values.size() in expected_sizes
    ), f"Values Tensor is expected to be of these sizes: {expected_sizes}"

    # Assert that indices has only one dimension
    assert (
        indices.ndim == 1
    ), f"Indices tensor must be 1-dimensional, but got {indices.ndim} dimensions"

    # Create a mask tensor with the same shape as sparse_tensor
    filled_indices = t.arange(sparse_tensor.size(0))
    filled_indices = filled_indices[
        ~t.isin(filled_indices, indices.cpu())
        & t.isin(filled_indices, sparse_tensor.indices()[0].cpu())
    ]
    if filled_indices.numel() == 0:
        mask = t.zeros(
            sparse_tensor.size(),
            dtype=sparse_tensor.dtype,
            device=sparse_tensor.device,
            layout=sparse_tensor.layout,
        )
    else:
        mask = _create_sparse_mask(
            sparse_tensor.size(),
            filled_indices,
            t.tensor(1),
            dtype=sparse_tensor.dtype,
            device=sparse_tensor.device,
        )

    values = _create_sparse_mask(
        sparse_tensor.size(), indices, values, device=sparse_tensor.device
    )

    # Multiply the original sparse tensor by the mask (zeroing out the specified indices)
    sparse_tensor = sparse_tensor * mask
    # Add the new values at the specified indices
    sparse_tensor += values

    return sparse_tensor


def _create_sparse_mask(
    size: t.Size,
    indices: t.Tensor,
    fill: t.Tensor,
    dtype: t.dtype | None = None,
    device: t.device | None = None,
) -> t.Tensor:
    """
    Create a sparse mask tensor filled with `fill` at the specified `indices` at dim=0.
    Sparse equivalent to `t.zeros(size).index_put_((indices,) fill)`.

    Args:
        size: Target tensor size
        indices: Indices along dim=0 where to place values
        fill: Values to place at indices. Can be:
            - single value: expanded to all positions
            - shape matching size[1:]: repeated for each index
            - arbitrary tensor: used directly for non-zero positions
        dtype: Optional dtype for output tensor
        device: Optional device for output tensor
    """
    # Handle single value fill case efficiently
    if fill.numel() == 1:
        if len(size) == 1:
            # For 1D case, just repeat the value for each index
            new_indices = indices.unsqueeze(0)
            fill_values = fill.repeat(indices.size(0))
        else:
            # For N-D case, create cartesian product only for non-zero fill
            new_indices = t.cartesian_prod(
                indices.cpu(), *[t.arange(dim) for dim in size[1:]]
            ).mT
            fill_values = fill.repeat(new_indices.size(1))
    else:
        # Get positions of non-zero elements in fill tensor
        nonzero_idxs = fill.nonzero().cpu()

        if size[1:] == fill.shape:
            # If fill matches trailing dimensions, repeat for each index
            new_indices = t.cat(
                [
                    indices.repeat_interleave(nonzero_idxs.size(0)).unsqueeze(-1),
                    nonzero_idxs.repeat(indices.size(0), 1),
                ],
                dim=1,
            ).mT
            fill_values = fill[tuple(nonzero_idxs.T)].repeat(indices.size(0))
        elif indices.size(0) == fill.size(0):
            # If first dimension matches indices count, use fill directly
            new_indices = nonzero_idxs.clone()
            for old_idx, new_idx in enumerate(indices):
                new_indices[nonzero_idxs[:, 0] == old_idx, 0] = new_idx
            # repeats = (
            #     nonzero_idxs[:, 0].unsqueeze(1) == t.arange(indices.size(0))
            # ).sum(dim=0)
            # indices = t.repeat_interleave(indices.unsqueeze(-1), repeats)
            # new_indices[:, 0] = indices
            new_indices = new_indices.mT
            fill_values = fill[tuple(nonzero_idxs.T)]
        else:
            raise ValueError(
                f"Invalid fill shape {fill.shape} for indices {indices.shape} and size {size}"
            )

    return t.sparse_coo_tensor(
        indices=new_indices,
        values=fill_values,
        size=size,
        dtype=dtype or fill.dtype,
        device=device or fill.device,
    ).coalesce()
