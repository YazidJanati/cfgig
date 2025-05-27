import torch
from tqdm import tqdm


def bridge_statistics(x_t, x_0, sigma_t, sigma_t_prev, eta):
    std = eta * (sigma_t_prev * (sigma_t**2.0 - sigma_t_prev**2.0).sqrt() / sigma_t)
    coeff_residue = (sigma_t_prev**2 - std**2).sqrt() / sigma_t
    mean = x_0 + coeff_residue * (x_t - x_0)

    return mean, std


def ddim_step(x_t, denoiser_fn, ctx, sigma_t, sigma_t_prev, eta):
    pred_x0 = denoiser_fn(x=x_t, sigma=sigma_t, ctx=ctx)
    mean, std = bridge_statistics(x_t, pred_x0, sigma_t, sigma_t_prev, eta)
    return mean + std * torch.randn_like(mean)


def ddim(
    initial_noise,
    denoiser_fn,
    ctx,
    mk_sigmas_fn,
    n_steps,
    sigma_min=0.0,
    sigma_max=1e4,
    eta=0.0,
    rho=7.0,
    init_xt=None,
    return_intermediate=False,
    **kwargs,
):
    """
    DDIM for variance exploding.
    """
    device = initial_noise.device if initial_noise is not None else init_xt.device

    sigmas = mk_sigmas_fn(
        n_steps=n_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
    ).to(device)
    sigmas_flipped = sigmas.flip(0)
    pbar = tqdm(zip(sigmas_flipped[:-1], sigmas_flipped[1:]))

    x_t = init_xt if init_xt is not None else sigmas[-1] * initial_noise
    if return_intermediate:
        arr_intermediate = [x_t.cpu()]

    for sigma_t, sigma_t_prev in pbar:
        x_t = ddim_step(x_t, denoiser_fn, ctx, sigma_t, sigma_t_prev, eta)

        if return_intermediate:
            arr_intermediate.append(x_t.cpu())

    if return_intermediate:
        return x_t, arr_intermediate

    return denoiser_fn(x_t, sigma_t_prev, ctx)


def heun(
    initial_noise,
    denoiser_fn,
    ctx,
    mk_sigmas_fn,
    n_steps,
    # ---
    sigma_min,
    sigma_max,
    rho=7.0,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    init_xt=None,
    **kwargs,
):
    """EDM Heun based Sampler with (optional) Limited interval Classifier Free Guidance."""

    sigmas = mk_sigmas_fn(
        n_steps=n_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
    )
    sigmas = sigmas.flip(0)
    sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])  # t_N = 0
    pbar = tqdm(enumerate(zip(sigmas[:-1], sigmas[1:])))
    # x_next = initial_noise * sigmas[0]
    x_next = init_xt if init_xt is not None else sigmas[0] * initial_noise
    for i, (t_cur, t_next) in pbar:  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = (
            min(S_churn / n_steps, 2 ** (1 / 2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised = denoiser_fn(x_hat, t_hat, ctx)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < n_steps - 1:
            denoised = denoiser_fn(x_next, t_next, ctx)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next
