import warnings
from typing import List

import torch
from torch import nn

from typing import Tuple, Union

from edm2 import EDM_MODELS, load_edm2
from local_paths import REPO_PATH

import torch
from torch import nn
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline, AudioLDM2Pipeline


def load_sd_epsilon_net(model_id, dtype=torch.float32):

    if model_id == "sd1.5":
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=dtype
        )
    elif model_id == "sd2.1":
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", torch_dtype=dtype
        )
    elif model_id == "sdxl1.0":
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype
        )
    else:
        raise ValueError("wrong model_id or not supported")

    return pipeline


def load_al_epsilon_net(model_id, dtype=torch.float32):
    if model_id == "audioldm2":
        pipeline = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2", torch_dtype=dtype
        )
    elif model_id == "audioldm2-large":
        pipeline = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2-large", torch_dtype=dtype
        )
    elif model_id == "audioldm2-music":
        pipeline = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2-music", torch_dtype=dtype
        )
    elif model_id == "audioldm2-ljspeech":
        pipeline = AudioLDM2Pipeline.from_pretrained(
            "anhnct/audioldm2_ljspeech", torch_dtype=dtype
        )
    elif model_id == "audioldm2-gigaspeech":
        pipeline = AudioLDM2Pipeline.from_pretrained(
            "anhnct/audioldm2_gigaspeech", torch_dtype=dtype
        )
    else:
        raise ValueError("wrong model_id or not supported")

    return pipeline


# ---
# NOTE: reverse engineered from pipeline().image in diffusers v0.27.2
# where pipeline is instance of `StableDiffusionXLPipeline`
def _get_cond_kwargs(other_prompt_embeds: List[torch.Tensor], use_cfg: bool):
    # case of sd1.5 and sd2.1
    if other_prompt_embeds is None or len(other_prompt_embeds) == 0:
        return

    # otherwise, this handles the case of stable diffusion xl-base-1.0
    pooled_prompt_embeds, negative_pooled_prompt_embeds = other_prompt_embeds

    # additional image-based embeddings
    add_time_ids = _get_time_ids(pooled_prompt_embeds)
    negative_add_time_ids = add_time_ids

    add_text_embeds = pooled_prompt_embeds
    if use_cfg:
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds]
        )
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids])

    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    return added_cond_kwargs


def _get_time_ids(prompt_embeds: torch.Tensor):
    # NOTE these were deduced from pipeline.__call__ of diffuser v0.27.2
    # and are so far valid for sdxl1.0
    original_size = (1024, 1024)
    crops_coords_top_left = (0, 0)
    target_size = (1024, 1024)

    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor(
        prompt_embeds.shape[0] * [add_time_ids], dtype=prompt_embeds.dtype
    )
    return add_time_ids


# ---
class EDM2Wrapper(nn.Module):

    def __init__(self, cond_model, uncond_model=None, vae=None):
        super().__init__()

        self.cond_model = cond_model
        self.uncond_model = uncond_model
        self.vae = vae

        self.x_shape = (
            cond_model.img_channels,
            cond_model.img_resolution,
            cond_model.img_resolution,
        )

    def denoiser_fn(self, x, sigma, ctx, use_cfg: bool = True):
        cond_model = self.cond_model
        uncond_model = self.uncond_model
        n_samples = x.shape[0]

        # NOTE handle many samples
        # here, we divide by `len(ctx)` as `n_samples` represent
        # the total number of samples
        ctx = self.make_ctx(ctx)
        ctx = ctx.repeat_interleave(n_samples // len(ctx), dim=0)

        # preprocess sigma to be passed in to the denoiser
        # handle the case where the denoiser is applied to different times steps
        if len(sigma.shape) == 0:
            sigmas = sigma * torch.ones((n_samples, 1))
        elif sigma.shape[0] != n_samples:
            raise ValueError(
                "parameter `sigma` must be a scaler or have the same size as `x`"
            )
        else:
            sigmas = sigma

        # Apply without Classifier Free Guidance
        if not use_cfg:
            x0s = cond_model(x, sigmas, ctx)
            return x0s

        # Classifier Free Guidance
        if uncond_model is None:

            x0s = cond_model(
                torch.cat([x] * 2),
                torch.cat([sigmas] * 2),
                torch.cat([torch.zeros_like(ctx), ctx]),
            )
            return x0s

        # Guidance with an auxiliary network
        else:
            x0_cond = cond_model(x, sigmas, ctx)
            x0_uncond = uncond_model(x, sigmas, torch.zeros_like(ctx))
            return torch.cat([x0_uncond, x0_cond])

    def make_ctx(self, ctx: List[int]):
        class_label = torch.nn.functional.one_hot(
            torch.tensor(ctx), num_classes=self.cond_model.label_dim
        )
        return class_label

    def mk_sigmas_fn(
        self,
        n_steps: int,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
    ) -> torch.Tensor:
        sigmas = (
            sigma_max ** (1 / rho)
            + torch.linspace(1, 0, n_steps)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        return sigmas

    def decode(self, x):
        return self.vae.decode(x) if self.vae is not None else x

    def encode(self, x):
        return self.vae.encode(x) if self.vae is not None else x

    # NOTE: this is for compatibility with torch NN
    def forward(self, x, sigma, ctx):
        return self.denoiser_fn(x, sigma, ctx)


class SDWrapper(nn.Module):

    def __init__(
        self, model_id, device, cache_prompt_embeddings=True, dtype=torch.float32
    ):
        super().__init__()
        self.dtype = dtype
        self.pipeline = load_sd_epsilon_net(model_id, dtype).to(device)
        self.unet = self.pipeline.unet

        if model_id == "sd2.1":
            self.prediction_type = "v_prediction"

        elif model_id in ["sd1.5", "sdxl1.0"]:
            self.prediction_type = "noise_prediction"

        self.vae = self.pipeline.vae
        self.scheduler = self.pipeline.scheduler
        self.acp = self.scheduler.alphas_cumprod.to(device)
        self.sigmas = (1 - self.acp).sqrt() / self.acp.sqrt()

        self.acp = self.acp.to(dtype)
        self.sigmas = self.sigmas.to(dtype)

        self.channels = self.pipeline.unet.config.in_channels
        self.im_size = self.pipeline.unet.config.sample_size

        # for caching
        # XXX caching was manully implemented to control cache side-effects
        self.cache_prompt_embeddings = cache_prompt_embeddings
        self._cache = {"prompt": None, "prompt_embbedding": None, "use_cfg": None}

        self.x_shape = (self.channels, self.im_size, self.im_size)

    @torch.no_grad
    def denoiser_fn(self, x, sigma, ctx, use_cfg: bool = True):
        # preprocess sigma
        # handle the case where the denoiser is applied to different times steps
        if len(sigma.shape) == 0:
            sigmas = sigma.view(1)
        elif sigma.shape[0] != x.shape[0]:
            raise ValueError(
                "argument `sigma` must be a scaler or have the same `shape[0]` as `x`"
            )
        else:
            sigmas = sigma

        timestep = (1.0 * (sigmas[:, None] < self.sigmas)).argmax(dim=1).to(self.dtype)
        x_vp = x / (1 + sigmas**2).sqrt().view(-1, *[1 for _ in self.x_shape])
        return self._vp_denoiser_fn(x_vp, timestep, ctx, use_cfg)

    def _vp_denoiser_fn(self, x, t, ctx, use_cfg: bool = False):
        """
        computes the x0 prediction
        """

        if (
            self.cache_prompt_embeddings
            and ctx == self._cache["prompt"]
            and use_cfg == self._cache["use_cfg"]
        ):
            prompt_embeds, uncond_prompt_embeds, *others = self._cache["embeddings"]
        else:
            prompt_embeds, uncond_prompt_embeds, *others = self.pipeline.encode_prompt(
                prompt=ctx,
                device=x.device,
                num_images_per_prompt=x.shape[0] // len(ctx),
                do_classifier_free_guidance=use_cfg,
            )
            if self.cache_prompt_embeddings:
                self._cache["embeddings"] = (
                    prompt_embeds,
                    uncond_prompt_embeds,
                    *others,
                )
                self._cache["prompt"] = ctx
                self._cache["use_cfg"] = use_cfg

        if use_cfg:
            prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds])
            x = torch.cat([x] * 2)

        added_cond_kwargs = _get_cond_kwargs(others, use_cfg)
        pred = self.unet(
            x,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        acp_t = self.acp[t.long()]
        acp_t = acp_t.view(-1, *[1 for _ in self.x_shape])

        if self.prediction_type == "v_prediction":
            pred_x0 = acp_t.sqrt() * x - (1 - acp_t).sqrt() * pred

        elif self.prediction_type == "noise_prediction":
            pred_x0 = (x - (1 - acp_t).sqrt() * pred) / acp_t.sqrt()

        return pred_x0

    def mk_sigmas_fn(self, n_steps, sigma_min, sigma_max, **kwargs):
        sigmas = self.sigmas[(sigma_min < self.sigmas) & (self.sigmas < sigma_max)]

        if n_steps > len(sigmas):
            warnings.warn(
                "`n_steps` is greater than the length of `sigmas` and hence results in"
                "duplicate `sigma_t` that might cause in instabilities during sampling"
            )

        idx = torch.linspace(0, len(sigmas) - 1, n_steps, dtype=torch.int32)
        return sigmas[idx]

    @torch.no_grad
    def decode(self, x, force_float32: bool = True):
        # force decoding to be in float32 to avoid overflow
        vae = self.vae.to(torch.float32) if force_float32 else self.vae
        x = x.to(torch.float32) if force_float32 else x

        return vae.decode(x / vae.config.scaling_factor).sample


class ALWrapper(nn.Module):

    def __init__(
        self,
        model_id,
        device,
        duration_s: int = 10,
        dtype=torch.float32,
        cache_prompt_embeddings: bool = True,
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.pipeline = load_al_epsilon_net(model_id, dtype).to(device)
        self.unet = self.pipeline.unet
        self.vocoder = self.pipeline.vocoder
        self.vae_scale_factor = self.pipeline.vae_scale_factor
        self.vae = self.pipeline.vae
        self.scheduler = self.pipeline.scheduler
        self.mel_to_wav = self.pipeline.mel_spectrogram_to_waveform
        self.score_wavs = self.pipeline.score_waveforms
        self.acp = (1.0 - self.scheduler.betas).cumprod(dim=0).to(device)
        self.sigmas = (1 - self.acp).sqrt() / self.acp.sqrt()
        self.acp = self.acp.to(dtype)
        self.sigmas = self.sigmas.to(dtype)
        self.channels = self.pipeline.unet.config.in_channels
        self.duration_s = duration_s

        # for caching
        # XXX caching was manully implemented to control cache side-effects
        self.cache_prompt_embeddings = cache_prompt_embeddings
        self._cache = {"prompt": None, "prompt_embbedding": None, "use_cfg": None}

        # TODO retrieve hard coded shape from pipeline __call__
        self.x_shape = (self.channels, 250, 16)

    @torch.no_grad
    def denoiser_fn(self, x, sigma, ctx, use_cfg: bool = True):
        # preprocess sigma
        # handle the case where the denoiser is applied to different times steps
        if len(sigma.shape) == 0:
            sigmas = sigma.view(1)
        elif sigma.shape[0] != x.shape[0]:
            raise ValueError(
                "argument `sigma` must be a scaler or have the same `shape[0]` as `x`"
            )
        else:
            sigmas = sigma

        timestep = (1.0 * (sigmas[:, None] < self.sigmas)).argmax(dim=1).to(self.dtype)
        x_vp = x / (1 + sigmas**2).sqrt().view(-1, *[1 for _ in self.x_shape])
        return self._vp_denoiser_fn(x_vp, timestep, ctx, use_cfg)

    def _vp_denoiser_fn(self, x, t, ctx, use_cfg: bool = False):
        """
        computes the x0 prediction
        """
        num_wavs_per_prompt = x.shape[0] // len(ctx)

        if (
            self.cache_prompt_embeddings
            and ctx == self._cache["prompt"]
            and use_cfg == self._cache["use_cfg"]
        ):
            prompt_embeds, attention_mask, generated_prompt_embeds = self._cache[
                "embeddings"
            ]
        else:
            prompt_embeds, attention_mask, generated_prompt_embeds = (
                self.pipeline.encode_prompt(
                    prompt=ctx,
                    negative_prompt=["Low quality."] * len(ctx),
                    transcription=None,
                    max_new_tokens=None,
                    device=self.device,
                    num_waveforms_per_prompt=num_wavs_per_prompt,
                    do_classifier_free_guidance=use_cfg,
                )
            )
            # save to cache if required
            if self.cache_prompt_embeddings:
                self._cache["embeddings"] = (
                    prompt_embeds,
                    attention_mask,
                    generated_prompt_embeds,
                )
                self._cache["prompt"] = ctx
                self._cache["use_cfg"] = use_cfg

        if use_cfg:
            x = torch.cat([x] * 2)

        pred = self.unet(
            x,
            t,
            # === cross-attn with GPT2 embeddings (256-d) ===
            encoder_hidden_states=generated_prompt_embeds,
            # === cross-attn with Flan-T5 embeddings (1024-d) ===
            encoder_hidden_states_1=prompt_embeds,
            encoder_attention_mask_1=attention_mask,
            return_dict=False,
        )[0]

        acp_t = self.acp[t.long()]
        acp_t = acp_t.view(-1, *[1 for _ in self.x_shape])
        # noise_prediction
        pred_x0 = (x - (1 - acp_t).sqrt() * pred) / acp_t.sqrt()
        return pred_x0

    def mk_sigmas_fn(self, n_steps, sigma_min, sigma_max, **kwargs):
        sigmas = self.sigmas[(sigma_min < self.sigmas) & (self.sigmas < sigma_max)]

        if n_steps > len(sigmas):
            warnings.warn(
                "`n_steps` is greater than the length of `sigmas` and hence results in"
                "duplicate `sigma_t` that might cause in instabilities during sampling"
            )

        idx = torch.linspace(0, len(sigmas) - 1, n_steps, dtype=torch.int32)
        return sigmas[idx]

    @torch.no_grad()
    def decode(self, x, ctx, *, select_best_audio: bool = True):
        num_wavs_per_prompt = x.shape[0] // len(ctx)
        x = 1 / self.vae.config.scaling_factor * x
        mel_spectrogram = self.vae.decode(x).sample
        mel_spectrogram = mel_spectrogram.to(self.dtype)
        audio = self.mel_to_wav(mel_spectrogram)
        original_waveform_length = int(
            self.duration_s * self.vocoder.config.sampling_rate
        )
        audio = audio[:, :original_waveform_length]

        if select_best_audio and num_wavs_per_prompt > 1 and ctx is not None:
            best_audio = torch.zeros(
                (len(ctx), original_waveform_length), dtype=audio.dtype
            )

            audio = audio.reshape(
                len(ctx), num_wavs_per_prompt, original_waveform_length
            )
            for i, audio_ctx in enumerate(audio):
                # this automatically score the audios in descending order
                audio_ctx = self.score_wavs(
                    text=ctx,
                    audio=audio_ctx,
                    num_waveforms_per_prompt=num_wavs_per_prompt,
                    device=self.device,
                    dtype=self.dtype,
                )
                best_audio[i] = audio_ctx[0]

            return best_audio.cpu().numpy()

        return audio.cpu().numpy()


def load_model(
    cond_model_id: str,
    device: str = "cpu",
) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
    """
    #TODO: available models
    """
    edm2_models = EDM_MODELS
    sd_models = ["sdxl1.0", "sd2.1", "sd1.5"]
    al_models = [
        "audioldm2",
        "audioldm2-large",
        "audioldm2-music",
        "audioldm2-ljspeech",
        "audioldm2-gigaspeech",
    ]

    if cond_model_id in edm2_models:
        cond_network_url = edm2_models[cond_model_id]["net"]
        guid_network_url = edm2_models[cond_model_id].get("gnet", None)

        model, encoder = load_edm2(network_url=cond_network_url, device=device)
        cond_model = prepare_model_to_eval(model, device=device)

        if guid_network_url is not None:
            uncond_model, *_ = load_edm2(network_url=guid_network_url, device=device)
            uncond_model = prepare_model_to_eval(uncond_model, device=device)
        else:
            uncond_model = None

        return EDM2Wrapper(cond_model, uncond_model, vae=encoder)

    elif cond_model_id in sd_models:
        return SDWrapper(model_id=cond_model_id, device=device, dtype=torch.float32)

    elif cond_model_id in al_models:
        return ALWrapper(model_id=cond_model_id, device=device, dtype=torch.float32)


def prepare_model_to_eval(model, device):
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model
