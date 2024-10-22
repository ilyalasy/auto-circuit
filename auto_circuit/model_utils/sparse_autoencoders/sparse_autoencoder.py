import einops
import torch.profiler
from jaxtyping import Float
from sae_lens.sae import SAE, SAEConfig
from transformer_lens.hook_points import HookPoint

CHUNK_SIZE = 1000 * 2**20  # 1000 MB


class SAEWrapper(SAE):
    def __init__(
        self,
        cfg: SAEConfig,
        use_error_term: bool = False,
    ):
        super().__init__(cfg, use_error_term)
        self.hook_sae_latents = HookPoint()
        self.setup()

    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """Decodes SAE feature activation tensor into a reconstructed input activation tensor."""

        ein_str = "... l, ... d l -> ... l d"
        # save individual feature contribution
        latent_outs = self.hook_sae_latents(
            einops.einsum(
                self.apply_finetuning_scaling_factor(feature_acts),
                self.W_dec.T,
                ein_str,
            )
        )
        # "... d_sae, d_sae d_in -> ... d_in",
        sae_out = self.hook_sae_recons(latent_outs.sum(dim=2) + self.b_dec)

        # handle run time activation normalization if needed
        # will fail if you call this twice without calling encode in between.
        sae_out = self.run_time_activation_norm_fn_out(sae_out)

        # handle hook z reshaping if needed.
        sae_out = self.reshape_fn_out(sae_out, self.d_head)  # type: ignore

        return sae_out


# class SparseAutoencoder(t.nn.Module):
#     """
#     A Sparse Autoencoder wrapper module.

#     Takes some input, passes it through the autoencoder and passes the reconstructed
#     input to the wrapped hook.

#     Implements:
#         latents = ReLU(encoder(x - bias) + latent_bias)
#         recons = decoder(latents) + bias
#     """

#     def __init__(self, wrapped_hook: HookPoint, n_latents: int, n_inputs: int) -> None:
#         """
#         :param wrapped_hook: the wrapped transformer_lens hook that caches the SAE input
#         :param n_latents: dimension of the autoencoder latent
#         :param n_inputs: dimensionality of the input (e.g residual stream, MLP neurons)
#         """
#         super().__init__()
#         self.wrapped_hook: HookPoint = wrapped_hook
#         self.latent_outs: HookPoint = HookPoint()
#         # Weights start the same at each position. They're only different after pruning.
#         self.init_params(n_latents, n_inputs)
#         self.reset_activated_latents()

#     def init_params(
#         self, n_latents: int, n_inputs: int, seq_len: Optional[int] = None
#     ) -> None:
#         self.n_latents: int = n_latents
#         self.n_inputs: int = n_inputs
#         self.bias: t.nn.Parameter = t.nn.Parameter(t.zeros(n_inputs))
#         seq_shape = [] if seq_len is None else [seq_len]
#         self.latent_bias = t.nn.Parameter(t.zeros(seq_shape + [n_latents]))
#         self.encode_weight = t.nn.Parameter(t.zeros(seq_shape + [n_latents, n_inputs]))
#         self.decode_weight = t.nn.Parameter(t.zeros(seq_shape + [n_inputs, n_latents]))
#         [kaiming_uniform_(w) for w in [self.encode_weight, self.decode_weight]]

#     def reset_activated_latents(
#         self, batch_len: Optional[int] = None, seq_len: Optional[int] = None
#     ):
#         device = self.bias.device
#         batch_shape = [] if batch_len is None else [batch_len]
#         seq_shape = [] if seq_len is None else [seq_len]
#         shape = batch_shape + seq_shape + [self.n_latents]
#         self.register_buffer("latent_total_act", t.zeros(shape, device=device), False)

#     @classmethod
#     def from_state_dict(
#         cls, wrapped_hook: HookPoint, state_dict: Dict[str, t.Tensor]
#     ) -> "SparseAutoencoder":
#         n_latents, n_inputs = state_dict["encode_weight"].shape
#         autoencoder = cls(wrapped_hook, n_latents, n_inputs)
#         autoencoder.load_state_dict(state_dict, strict=True, assign=True)
#         autoencoder.reset_activated_latents()
#         return autoencoder

#     def prune_latents(self, idxs: t.Tensor):
#         assert idxs.ndim <= 2
#         state = self.state_dict()
#         assert state["encode_weight"].ndim == 2

#         new_state_dict = {
#             "bias": state["bias"].clone(),
#             "encode_weight": state["encode_weight"][idxs].clone(),
#             "latent_bias": state["latent_bias"][idxs].clone(),
#             "decode_weight": state["decode_weight"].T[idxs].transpose(-1, -2).clone(),
#         }
#         del state
#         self.init_params(*list(reversed(new_state_dict["decode_weight"].shape)))
#         self.load_state_dict(new_state_dict, assign=True, strict=True)
#         self.reset_activated_latents()

#     def encode(self, x: t.Tensor) -> t.Tensor:
#         """
#         :param x: input data (shape: [..., [seq], n_inputs])
#         :return: autoencoder latents (shape: [..., [seq], n_latents])
#         """

#         encoded = einsum(x - self.bias, self.encode_weight, "... d, ... l d -> ... l")
#         latents_pre_act = encoded + self.latent_bias
#         return t.nn.functional.relu(latents_pre_act)

#     def decode(self, x: t.Tensor) -> t.Tensor:
#         """
#         :param x: autoencoder x (shape: [..., [seq], n_latents])
#         :return: reconstructed data (shape: [..., [seq], n_inputs])
#         """

#         ein_str = "... l, ... d l -> ... l d"
#         latent_outs = self.latent_outs(einsum(x, self.decode_weight, ein_str))
#         return latent_outs.sum(dim=-2) + self.bias

#     def forward(self, x: t.Tensor) -> t.Tensor:
#         """
#         :param x: input data (shape: [..., n_inputs])
#         :return:  reconstructed data (shape: [..., n_inputs])
#         """
#         # with torch.profiler.profile(
#         #     profile_memory=True,
#         #     record_shapes=True,
#         #     with_stack=True,
#         #     on_trace_ready=trace_handler,
#         # ) as prof:
#         x = self.wrapped_hook(x)
#         # with torch.profiler.record_function("encode"):
#         latents = self.encode(x)
#         self.latent_total_act += latents.sum_to_size(self.latent_total_act.shape)
#         # with torch.profiler.record_function("decode"):
#         recons = self.decode(latents)
#         # prof.export_memory_timeline("sae_forward.html")
#         # prof.export_chrome_trace("sae_forward_trace.json")
#         return recons


STATE_DICT_MAPS = {
    "gpt2": {
        "pre_bias": "bias",
        "encoder.weight": "encode_weight",
        "latent_bias": "latent_bias",
        "decoder.weight": "decode_weight",
    },
    "pythia-70m-deduped": {
        "bias": "bias",
        "encoder.weight": "encode_weight",
        "encoder.bias": "latent_bias",
        "decoder.weight": "decode_weight",
    },
}


# def load_autoencoder(
#     wrapped_hook: HookPoint,
#     model: HookedTransformer,
#     layer_idx: int,
#     autoencoder_input: AutoencoderInput,
#     pythia_size: Optional[str] = None,
# ) -> SparseAutoencoder:
#     model_name = model.cfg.model_name
#     cache_dir = repo_path_to_abs_path(".autoencoder_cache")
#     if model_name == "gpt2":
#         blob_prefix = "az://openaipublic/sparse-autoencoder/gpt2-small"
#         blobpath = f"{blob_prefix}/{autoencoder_input}/autoencoders/{layer_idx}.pt"
#         cache_path = cache_dir / f"gpt2_{autoencoder_input}_{layer_idx}.pt"
#         if not cache_path.exists():
#             load_details = f"{autoencoder_input}, layer {layer_idx} to {cache_path}"
#             print(f"Downloading autoencoder {load_details}...")
#             start_time = time()
#             with BlobFile(blobpath, "rb") as blob, open(cache_path, "wb") as cache_file:
#                 block = blob.read(CHUNK_SIZE)
#                 cache_file.write(block)
#             # bf.copy(blobfile_path, str(cache_path))
#             print(f"Done. Took {time() - start_time:.2f} seconds.")
#     elif model_name == "pythia-70m-deduped":
#         assert pythia_size is not None and autoencoder_input == "resid_delta_mlp"
#         base_url = "https://baulab.us/u/smarks/autoencoders/pythia-70m-deduped/"
#         file_url = base_url + f"mlp_out_layer{layer_idx}/{pythia_size}/ae.pt"
#         file_name = f"pythia-70m-deduped_layer_{layer_idx}_{pythia_size}.pt"
#         cache_path = cache_dir / file_name
#         t.hub.load_state_dict_from_url(
#             file_url,
#             str(cache_dir.resolve()),
#             file_name=file_name,
#             map_location=model.cfg.device,
#         )
#     else:
#         raise ValueError(f"No autoencoder support for model {model_name}")

#     with open(cache_path, "rb") as f:
#         state_dict = t.load(f, map_location=model.cfg.device)
#     map_dict: Dict[str, str] = STATE_DICT_MAPS[model_name]
#     state_dict = {map_dict[k]: v for k, v in state_dict.items() if k in map_dict}
#     return SparseAutoencoder.from_state_dict(wrapped_hook, state_dict)


import socket
from datetime import datetime

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


def trace_handler(prof: torch.profiler.profile):
    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    # Construct the trace file.
    prof.export_chrome_trace(f"profiler/{file_prefix}.json")

    # Construct the memory timeline file.
    prof.export_memory_timeline(f"profiler/{file_prefix}.html")  # device="mps"
