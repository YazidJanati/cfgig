import torch
from typing import Callable


def cfg_denoiser(
    denoiser_fn: Callable,
    cfg_cond_fn: Callable[[float, float], bool],
    cfg_scale: float,
):

    def cfg_denoiser_fn(x, sigma, ctx=None):
        if cfg_cond_fn(sigma, cfg_scale):
            pred_x0 = denoiser_fn(
                x=x,
                sigma=sigma,
                ctx=ctx,
                use_cfg=True,
            )
            predx0_uncond, predx0_cond = pred_x0.chunk(2)
            pred_x0 = predx0_cond + (cfg_scale - 1) * (predx0_cond - predx0_uncond)

        else:
            pred_x0 = denoiser_fn(x=x, sigma=sigma, ctx=ctx, use_cfg=False)
        return pred_x0

    return cfg_denoiser_fn


def cfg_sampler(
    initial_noise: torch.Tensor,
    denoiser_fn: Callable,
    ctx: torch.Tensor,
    mk_sigmas_fn: Callable[[int, float, float], torch.Tensor],
    n_steps: int,
    cfg_scale: float,
    sampler: Callable,
    **sampler_kwargs,
):

    cfg_denoiser_fn = cfg_denoiser(
        denoiser_fn,
        cfg_cond_fn=lambda sigma, scale: scale > 1.0,
        cfg_scale=cfg_scale,
    )

    return sampler(
        initial_noise=initial_noise,
        denoiser_fn=cfg_denoiser_fn,
        ctx=ctx,
        mk_sigmas_fn=mk_sigmas_fn,
        n_steps=n_steps,
        **sampler_kwargs,
    )


def cfgig_sampler(
    initial_noise: torch.Tensor,
    denoiser_fn: Callable,
    ctx: torch.Tensor,
    mk_sigmas_fn: Callable[[int, float, float], torch.Tensor],
    init_nsteps: int,
    n_steps: int,
    cfg_scale: float,
    sampler: Callable,
    sigma_gibbs: float,
    n_gibbs: int,
    cfg_scale_init: float = torch.tensor(1.0),
    keep_intermediates: bool = False,
    **sampler_kwargs,
):

    init_nsteps = min(n_steps, init_nsteps)
    gibbs_steps, rem = (
        (0, n_steps - init_nsteps)
        if n_gibbs == 0
        else divmod(n_steps - init_nsteps, n_gibbs)
    )

    x_0 = sampler(
        initial_noise,
        denoiser_fn=cfg_denoiser(
            denoiser_fn=denoiser_fn,
            cfg_cond_fn=lambda sigma, scale: scale > 1.0,
            cfg_scale=cfg_scale_init,
        ),
        ctx=ctx,
        mk_sigmas_fn=mk_sigmas_fn,
        n_steps=init_nsteps + rem,
        **sampler_kwargs,
    )

    if keep_intermediates:
        x0s = [x_0]

    sampler_kwargs["sigma_max"] = sigma_gibbs

    for _ in range(n_gibbs):

        x_t = x_0 + sigma_gibbs * torch.randn_like(x_0)

        x_0 = sampler(
            initial_noise=None,
            init_xt=x_t,
            denoiser_fn=cfg_denoiser(
                denoiser_fn=denoiser_fn,
                cfg_cond_fn=lambda sigma, scale: scale > 1.0,
                cfg_scale=cfg_scale,
            ),
            mk_sigmas_fn=mk_sigmas_fn,
            ctx=ctx,
            n_steps=gibbs_steps,
            **sampler_kwargs,
        )

        if keep_intermediates:
            x0s.append(x_0)

    return x0s if keep_intermediates else x_0


def limitedcfg_sampler(
    initial_noise: torch.Tensor,
    denoiser_fn: Callable,
    ctx: torch.Tensor,
    mk_sigmas_fn: Callable[[int, float, float], torch.Tensor],
    n_steps: int,
    cfg_scale: float,
    sampler: Callable,
    sigma_lo: float = 0.19,
    sigma_hi: float = 3.0,
    **sampler_kwargs,
):

    cfg_denoiser_fn = cfg_denoiser(
        denoiser_fn,
        cfg_cond_fn=lambda sigma, scale: (scale > 1.0) & (sigma_lo < sigma < sigma_hi),
        cfg_scale=cfg_scale,
    )

    return sampler(
        initial_noise=initial_noise,
        denoiser_fn=cfg_denoiser_fn,
        ctx=ctx,
        mk_sigmas_fn=mk_sigmas_fn,
        n_steps=n_steps,
        **sampler_kwargs,
    )


def cfgpp_sampler(
    initial_noise: torch.Tensor,
    denoiser_fn: Callable,
    ctx: torch.Tensor,
    mk_sigmas_fn: Callable[[int, float, float], torch.Tensor],
    n_steps: int,
    cfg_scale: float,
    sampler: Callable,
    **sampler_kwargs,
):
    # NOTE here we use the equivalence between cfg++ and cfg
    # cfg++ can be seen as cfg with dynamic cfg_scale

    sigma_min, sigma_max, rho = (
        sampler_kwargs["sigma_min"],
        sampler_kwargs["sigma_max"],
        sampler_kwargs["rho"],
    )

    sigmas = mk_sigmas_fn(
        n_steps=n_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
    )
    sigmas = torch.cat([torch.zeros_like(sigmas[:1]), sigmas])

    def cfgpp_denoiser_fn(x, sigma, ctx):
        # NOTE when sigma equals `sigma_min`,
        # `sigma_prev` will equal 0 as we prepended 0 to `sigmas`
        sigma_prev = sigmas[sigmas < sigma][-1]
        cfgpp_scale = cfg_scale * sigma / (sigma - sigma_prev)
        return cfg_denoiser(
            denoiser_fn,
            cfg_cond_fn=lambda sigma, scale: scale > 0.0,
            cfg_scale=cfgpp_scale,
        )(x, sigma, ctx)

    return sampler(
        initial_noise=initial_noise,
        denoiser_fn=cfgpp_denoiser_fn,
        ctx=ctx,
        mk_sigmas_fn=mk_sigmas_fn,
        n_steps=n_steps,
        **sampler_kwargs,
    )


def delaycfg_sampler(
    initial_noise: torch.Tensor,
    denoiser_fn: Callable,
    ctx: torch.Tensor,
    mk_sigmas_fn: Callable[[int, float, float], torch.Tensor],
    init_nsteps: int,
    n_steps: int,
    cfg_scale: float,
    sampler: Callable,
    sigma_gibbs: float,
    n_gibbs: int,
    delta: float,
    cfg_scale_init: float = torch.tensor(1.0),
    keep_intermediates: bool = False,
    **sampler_kwargs,
):
    init_nsteps = min(n_steps, init_nsteps)
    gibbs_steps, rem = (
        (0, n_steps - init_nsteps)
        if n_gibbs == 0
        else divmod(n_steps - init_nsteps, n_gibbs)
    )

    def delay_denoiser_fn(x, sigma, ctx=None):
        predx0_1, predx0_2 = (
            denoiser_fn(
                x, (cfg_scale / (1 + delta)) ** 0.5 * sigma, ctx, use_cfg=False
            ),
            denoiser_fn(
                x,
                ((cfg_scale - 1) / delta) ** 0.5 * sigma,
                torch.zeros_like(ctx),
                use_cfg=False,
            ),
        )
        pred_x0 = predx0_1 + (cfg_scale - 1) * (predx0_1 - predx0_2)
        return pred_x0

    x_0 = sampler(
        initial_noise,
        denoiser_fn=cfg_denoiser(
            denoiser_fn=denoiser_fn,
            cfg_cond_fn=lambda sigma, scale: scale > 1.0,
            cfg_scale=cfg_scale_init,
        ),
        ctx=ctx,
        mk_sigmas_fn=mk_sigmas_fn,
        n_steps=init_nsteps + rem,
        **sampler_kwargs,
    )

    if keep_intermediates:
        x0s = [x_0]

    sampler_kwargs["sigma_max"] = sigma_gibbs

    for _ in range(n_gibbs):

        x_t = x_0 + sigma_gibbs * torch.randn_like(x_0)

        x_0 = sampler(
            initial_noise=None,
            init_xt=x_t,
            denoiser_fn=delay_denoiser_fn,
            mk_sigmas_fn=mk_sigmas_fn,
            ctx=ctx,
            n_steps=gibbs_steps,
            **sampler_kwargs,
        )

        if keep_intermediates:
            x0s.append(x_0)

    return x0s if keep_intermediates else x_0
