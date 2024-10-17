# %%

import pytest
import torch as t

from auto_circuit.data import PromptPairBatch
from auto_circuit.utils.tensor_ops import (
    assign_sparse_tensor,
    batch_answer_diff_percents,
    batch_avg_answer_diff,
    batch_avg_answer_val,
    correct_answer_greater_than_incorrect_proportion,
    correct_answer_proportion,
)


def test_batch_avg_answer_val():
    """
    Tests batch_avg_answer_val, which calculates the average value of the correct
    answer's logits.
    """
    logits = t.tensor([[0.75, 0.2, 0.7], [0.3, 0.25, 0.5]])
    batch = PromptPairBatch(
        key=12,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=t.tensor([[0], [1]]),
        wrong_answers=t.tensor([[2], [2]]),  # Not used in this test
    )
    assert batch_avg_answer_val(logits, batch).item() == 0.5
    list_batch = PromptPairBatch(
        key=12,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=[t.tensor([0]), t.tensor([1])],
        wrong_answers=[t.tensor([2]), t.tensor([2])],  # Not used in this test
    )
    assert batch_avg_answer_val(logits, list_batch).item() == 0.5


def test_batch_avg_answer_diff():
    """
    Tests batch_avg_answer_diff, which calculates the average difference between the
    correct and wrong answers' logits.
    """
    logits = t.tensor([[0.75, 0.2, 0.75], [0.3, 1.25, 0.25]])
    batch = PromptPairBatch(
        key=12,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=t.tensor([[0], [1]]),
        wrong_answers=t.tensor([[2], [2]]),  # Used in this test!
    )
    # Average of (0.75 - 0.75) and (1.25 - 0.25) is 0.5
    assert batch_avg_answer_diff(logits, batch).item() == 0.5
    list_batch = PromptPairBatch(
        key=12,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=[t.tensor([0]), t.tensor([1])],
        wrong_answers=[t.tensor([2]), t.tensor([2])],  # Used in this test!
    )
    assert batch_avg_answer_diff(logits, list_batch).item() == 0.5


def test_correct_answer_proportion():
    correct_logits = t.tensor([[1.0, 0.2, 0.7], [0.3, 0.9, 0.5]])
    half_correct_logits = t.tensor([[0.5, 0.2, 0.7], [0.3, 0.6, 0.5]])
    incorrect_logits = t.tensor([[0.1, 0.2, 0.7], [0.3, 0.2, 0.5]])
    batch = PromptPairBatch(
        key=12,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=t.tensor([[0], [1]]),
        wrong_answers=t.tensor([[2], [2]]),  # Not used in this test
    )
    assert correct_answer_proportion(correct_logits, batch).item() == 1.0
    assert correct_answer_proportion(half_correct_logits, batch).item() == 0.5
    assert correct_answer_proportion(incorrect_logits, batch).item() == 0.0
    list_prompt = PromptPairBatch(
        key=12,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=[t.tensor([0]), t.tensor([1])],
        wrong_answers=[t.tensor([2]), t.tensor([2])],  # Not used in this test
    )
    assert correct_answer_proportion(correct_logits, list_prompt).item() == 1.0
    assert correct_answer_proportion(half_correct_logits, list_prompt).item() == 0.5
    assert correct_answer_proportion(incorrect_logits, list_prompt).item() == 0.0


def test_correct_greater_than_incorrect_proportion():
    """
    Tests correct_answer_greater_than_incorrect_proportion, which just checks if the
    correct answer has a higher value than all the wrong answers.
    """
    correct_logits = t.tensor([[9.0, 2.0, 7.0, 1.0], [3.0, 9.0, 5.0, 1.0]])
    correct_vs_wrong_ans_only_logits = t.tensor(
        [[6.0, 2.0, 7.0, 1.0], [3.0, 4.0, 5.0, 1.0]]
    )
    half_correct_logits = t.tensor([[5.0, 7.0, 7.0, 1.0], [3.0, 6.0, 5.0, 1.0]])
    half_correct_vs_wrong_ans_only_logits = t.tensor(
        [[5.0, 7.0, 7.0, 1.0], [3.0, 6.0, 9.0, 1.0]]
    )
    incorrect_logits = t.tensor([[1.0, 2.0, 7.0, 1.0], [3.0, 2.0, 5.0, 1.0]])
    batch = PromptPairBatch(
        key=12,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=t.tensor([[0], [1]]),
        wrong_answers=t.tensor([[1, 3], [0, 3]]),  # Used in this test!
    )

    # Create aliases for the functions with shorter names to improve readability
    corr_greater_prop = correct_answer_greater_than_incorrect_proportion
    corr_prop = correct_answer_proportion

    assert corr_greater_prop(correct_logits, batch).item() == 1.0
    assert corr_prop(correct_vs_wrong_ans_only_logits, batch).item() == 0.0
    assert corr_greater_prop(correct_vs_wrong_ans_only_logits, batch).item() == 1.0
    assert corr_greater_prop(half_correct_logits, batch).item() == 0.5
    assert corr_prop(half_correct_vs_wrong_ans_only_logits, batch).item() == 0.0
    assert corr_greater_prop(half_correct_vs_wrong_ans_only_logits, batch).item() == 0.5
    assert corr_greater_prop(incorrect_logits, batch).item() == 0.0
    PromptPairBatch(
        key=12,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=[t.tensor([0]), t.tensor([1])],
        wrong_answers=[t.tensor([1, 3]), t.tensor([0, 3])],  # Used in this test!
    )
    assert corr_greater_prop(correct_logits, batch).item() == 1.0
    assert corr_prop(correct_vs_wrong_ans_only_logits, batch).item() == 0.0
    assert corr_greater_prop(correct_vs_wrong_ans_only_logits, batch).item() == 1.0
    assert corr_greater_prop(half_correct_logits, batch).item() == 0.5
    assert corr_prop(half_correct_vs_wrong_ans_only_logits, batch).item() == 0.0
    assert corr_greater_prop(half_correct_vs_wrong_ans_only_logits, batch).item() == 0.5
    assert corr_greater_prop(incorrect_logits, batch).item() == 0.0


def test_batch_answer_diff_percent():
    pred_vals: t.Tensor = t.tensor([[0.3, 0.7], [0.6, 0.4]])
    target_vals: t.Tensor = t.tensor([[0.2, 0.8], [0.5, 0.3]])
    batch: PromptPairBatch = PromptPairBatch(
        key=1,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=t.tensor([[0], [0]]),
        wrong_answers=t.tensor([[1], [1]]),
    )
    logit_diff_percents = batch_answer_diff_percents(pred_vals, target_vals, batch)

    true_pred_logit_diffs = pred_vals[:, 0] - pred_vals[:, 1]
    true_target_logit_diffs = target_vals[:, 0] - target_vals[:, 1]
    true_logit_diff_percents = (true_pred_logit_diffs / true_target_logit_diffs) * 100
    assert t.allclose(logit_diff_percents, true_logit_diff_percents)


def test_assign_sparse_tensor():
    # Test case 1: Assigning values to existing indices
    dense_tensor = t.tensor(
        [
            [
                [1, 2, 3],
                [0, 0, 0],
                [0, 5.1, 6.5],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 0],
                [0, 2.2, 0],
            ],
        ]
    )
    sparse_tensor = dense_tensor.to_sparse()
    result = assign_sparse_tensor(
        sparse_tensor,
        slice(0, 1, None),
        t.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
                [0, 5.1, 6.5],
                [0, 0, 0],
            ],
        ),
    )
    dense_res = t.tensor(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [0, 5.1, 6.5],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 0],
                [0, 2.2, 0],
            ],
        ]
    )
    assert t.allclose(result.to_dense(), dense_res)

    # Test case 1.5: Assigning values to existing indices
    dense_tensor = t.tensor(
        [
            [1, 2, 3],
            [0, 0, 0],
            [0, 5.1, 6.5],
            [0, 0, 0],
            [4, 5, 6],
            [0, 2, 0],
            [0, 0, 0],
            [0, 2.2, 0],
        ],
    )
    sparse_tensor = dense_tensor.to_sparse()
    result = assign_sparse_tensor(
        sparse_tensor,
        slice(2, 6, None),
        t.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
                [0, 5.1, 6.5],
                [0, 0, 0],
            ],
        ),
    )
    dense_res = t.tensor(
        [
            [1, 2, 3],
            [0, 0, 0],
            [1, 2, 3],
            [4, 5, 6],
            [0, 5.1, 6.5],
            [0, 0, 0],
            [0, 0, 0],
            [0, 2.2, 0],
        ],
    )
    assert t.allclose(result.to_dense(), dense_res)

    # Test case 2: Assigning values to new indices
    sparse_tensor = t.sparse_coo_tensor(
        indices=t.tensor([[0, 2]]), values=t.tensor([1, 3]), size=(4,)
    )
    result = assign_sparse_tensor(sparse_tensor, t.tensor([1]), t.tensor([2]))
    result = assign_sparse_tensor(result, t.tensor([3]), t.tensor([4]))
    assert t.allclose(result.to_dense(), t.tensor([1, 2, 3, 4]))

    # Test case 3: Using slice for indices
    sparse_tensor = t.sparse_coo_tensor(
        indices=t.tensor([[0, 1, 2, 3]]), values=t.tensor([1, 2, 3, 4]), size=(5,)
    )
    result = assign_sparse_tensor(sparse_tensor, slice(1, 3), t.tensor([5, 6]))
    assert t.allclose(result.to_dense(), t.tensor([1, 5, 6, 4, 0]))

    # Test case 4: Assigning to empty sparse tensor
    tensor = t.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    sparse_tensor = tensor.to_sparse()
    result = assign_sparse_tensor(sparse_tensor, t.tensor([2]), t.tensor([1, 2, 3]))
    assert t.allclose(
        result.to_dense(), t.tensor([[0, 0, 0], [0, 0, 0], [1, 2, 3], [0, 0, 0]])
    )

    # Test case 5: Assigning to 2D sparse tensor
    sparse_tensor = t.sparse_coo_tensor(
        indices=t.tensor([[0, 1], [1, 0]]), values=t.tensor([1, 2]), size=(4, 2)
    )
    result = assign_sparse_tensor(sparse_tensor, t.tensor([1, 2]), t.tensor([3, 4]))
    assert t.allclose(result.to_dense(), t.tensor([[0, 1], [3, 4], [3, 4], [0, 0]]))

    # Test case 6: Assertion error
    sparse_tensor = t.sparse_coo_tensor(
        indices=t.tensor([[0, 1], [1, 0]]), values=t.tensor([1, 2]), size=(4, 4)
    )

    with pytest.raises(AssertionError):
        assign_sparse_tensor(sparse_tensor, t.tensor([1, 2]), t.tensor([3, 4]))


# test_batch_avg_answer_val()
# test_batch_avg_answer_diff()
# test_correct_answer_proportion()
# test_correct_vs_incorrect_answer_proportion()
# test_batch_answer_diff_percent()
