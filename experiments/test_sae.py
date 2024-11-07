import torch as t

from auto_circuit.data import PromptDataLoader, load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.metrics.official_circuits.circuits.ioi_official import ioi_true_edges
from auto_circuit.metrics.prune_metrics.answer_diff_percent import answer_diff_percent
from auto_circuit.metrics.prune_metrics.correct_answer_percent import (
    measure_correct_ans_percent,
)
from auto_circuit.model_utils.sparse_autoencoders.autoencoder_transformer import (
    sae_model,
)
from auto_circuit.prune import run_circuits
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import AblationType, Measurements, PatchType, PruneScores
from auto_circuit.utils.graph_utils import (
    edge_counts_util,
    load_patchable_model,
    patchable_model,
    prune_latents_with_dataset,
)
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import (
    correct_answer_greater_than_incorrect_proportion,
    correct_answer_proportion,
    prune_scores_threshold,
)


def find_circuits(
    model: PatchableModel,
    train_loader: PromptDataLoader,
    test_loader: PromptDataLoader,
    ablation_type: AblationType = AblationType.RESAMPLE,
    patch_type: PatchType = PatchType.TREE_PATCH,
):

    if ablation_type.mean_over_dataset:
        clean_corrupt = None
    else:
        clean_corrupt = "corrupt"
    prune_scores: PruneScores = mask_gradient_prune_scores(
        model=model,
        dataloader=train_loader,
        official_edges=None,
        grad_function="logit",
        answer_function="avg_diff",
        mask_val=0.0,
        ablation_type=ablation_type,
        clean_corrupt=clean_corrupt,
    )
    edge_count = edge_counts_util(model.edges, prune_scores=prune_scores)

    return prune_scores, run_circuits(
        model,
        test_loader,
        edge_count,
        prune_scores,
        ablation_type=ablation_type,
        patch_type=patch_type,
    )


def find_min_circuit(circuit_accuracy: Measurements, eps: float = 0.2):
    model_edge_count, base_acc = circuit_accuracy[-1]
    for edge_count, acc in circuit_accuracy:
        if acc and (abs(acc - base_acc) / base_acc) < eps:
            return edge_count, acc
    return model_edge_count, base_acc


def get_official_circuit(
    model: PatchableModel, test_loader: PromptDataLoader, tok_pos: bool
):
    edges = ioi_true_edges(
        model,
        word_idxs=test_loader.word_idxs,
        token_positions=tok_pos,
        seq_start_idx=test_loader.diverge_idx,
    )
    ps = model.circuit_prune_scores(edges)

    return ps, run_circuits(
        model=model,
        dataloader=test_loader,
        test_edge_counts=[len(edges)],
        prune_scores=ps,
        patch_type=PatchType.TREE_PATCH,
        ablation_type=AblationType.RESAMPLE,
        render_graph=False,
    )


def main():

    # Set up device
    if t.cuda.is_available():
        device = "cuda"
    elif t.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    # device = "cpu"  # t.device("cpu")
    # Load the model
    model_name = "pythia-70m-deduped"
    sae_release_name = "pythia-70m-deduped-mlp-sm"
    sae_layer_name = "blocks.{}.hook_mlp_out"

    # model_name = "gpt2"
    # sae_release_name = (
    #     "gpt2-small-mlp-tm"  # "gpt2-small-res-jb" # TODO: doesn't work with residuals
    # )
    # sae_layer_name = "blocks.{}.hook_mlp_out"

    # Load the model using load_tl_model from experiment_utils

    # Create the sparse autoencoder model
    model = sae_model(
        model_name,
        sae_release_name,
        sae_layer_name,
        device=device,
    )

    # model = load_tl_model(model_name, device=device)

    # Load the dataset
    # dataset_name = "datasets/ioi/ioi_vanilla_template_prompts.json"

    dataset_name = "datasets/ioi/ioi_ABBA_template_0_prompts.json"
    batch_size = 4
    dataset_size = 50
    train_dataloader, test_dataloader = load_datasets_from_json(
        model,
        repo_path_to_abs_path(dataset_name),
        device=device,
        prepend_bos=True,
        batch_size=batch_size,
        train_test_size=(8 * dataset_size, dataset_size),
        return_seq_length=False,
        shuffle=True,
        pad=True,
    )

    model = load_patchable_model(
        model,
        factorized=True,
        nodes_path="experiments/pruned_nodes_full_dataset.pt",
        slice_output="last_seq",
        separate_qkv=True,
        device=device,
    )

    # model = patchable_model(
    #     model,
    #     factorized=True,
    #     slice_output="last_seq",
    #     separate_qkv=True,
    #     device=device,
    # )
    # model = prune_latents_with_dataset(model, train_dataloader, None)
    # model.save("gpt2_pruned_nodes_full_dataset.pt")

    prune_scores, circuits_out = find_circuits(
        model,
        train_dataloader,
        test_dataloader,
        # ablation_type=AblationType.TOKENWISE_MEAN_CORRUPT,
    )

    (
        logit_diff_percent_mean,
        logit_diff_percent_std,
        logit_diff_percents,
    ) = answer_diff_percent(
        model,
        test_dataloader,
        circuits_out,
        prob_func="logits",
        diff_of_means=True,
    )

    circuit_accuracy_base = measure_correct_ans_percent(
        model, test_dataloader, circuits_out
    )
    best_edge_count, best_circuit_accuracy = find_min_circuit(logit_diff_percent_mean)
    threshold = prune_scores_threshold(prune_scores, best_edge_count).detach().item()
    print(best_circuit_accuracy)
    print(best_edge_count)
    print(threshold)


if __name__ == "__main__":
    main()
