import torch as t

from auto_circuit.data import PromptDataLoader, load_datasets_from_json
from auto_circuit.model_utils.sparse_autoencoders.autoencoder_transformer import (
    sae_model,
)
from auto_circuit.prune import run_circuits
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import AblationType, PatchType, PruneScores
from auto_circuit.utils.graph_utils import edge_counts_util, patchable_model
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.utils.patchable_model import PatchableModel


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


def main():

    # Set up device
    if t.cuda.is_available():
        device = t.device("cuda")
    elif t.backends.mps.is_available():
        device = t.device("mps")
    else:
        device = t.device("cpu")
    device = "cpu"  # t.device("cpu")
    # Load the model
    model_name = "pythia-70m-deduped"
    # Load the model using load_tl_model from experiment_utils

    # Create the sparse autoencoder model
    model = sae_model(
        model_name,
        "pythia-70m-deduped-mlp-sm",
        "blocks.{}.hook_mlp_out",
        device=device,
    )

    # Load the dataset
    dataset_name = "datasets/ioi/ioi_vanilla_template_prompts.json"
    batch_size = 2
    train_dataloader, test_dataloader = load_datasets_from_json(
        model,
        repo_path_to_abs_path(dataset_name),
        device=device,
        prepend_bos=True,
        batch_size=batch_size,
        train_test_size=(8, 8),
        return_seq_length=False,
        shuffle=True,
        pad=True,
    )

    model = patchable_model(
        model,
        factorized=True,
        slice_output="last_seq",
        separate_qkv=True,
        device=device,
    )
    prune_scores_ft, circuits_out_ft = find_circuits(
        model,
        train_dataloader,
        test_dataloader,
        # ablation_type=AblationType.TOKENWISE_MEAN_CORRUPT,
    )

    print(prune_scores_ft)
    print(circuits_out_ft)


if __name__ == "__main__":
    main()
