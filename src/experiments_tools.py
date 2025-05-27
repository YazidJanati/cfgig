import torchaudio


def update_sampler_config(config):
    """
    This function exists because hydra doesn't allow dynamic interpolation with nested files.
    The dataset/task specific parameters contained in add_cfg cannot be overridden with command line
    """
    sampler_ctx_params = getattr(config.sampler, "context_parameters", None)
    base_sampler_ctx_params = getattr(config.base_sampler, "context_parameters", None)

    # update sampler parameters
    if sampler_ctx_params is not None:
        params = getattr(sampler_ctx_params.model, config.model, None)
        if params is not None:
            config.sampler.parameters.update(params)

    if base_sampler_ctx_params is not None:
        base_params = getattr(base_sampler_ctx_params.model, config.model, None)
        if base_params is not None:
            config.base_sampler.parameters.update(base_params)


def save_audio(x, name, path, sample_rate):
    stream = x.detach().cpu().unsqueeze(0)
    torchaudio.save(path / f"{name}.wav", stream, sample_rate)
