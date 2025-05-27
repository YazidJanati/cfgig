from tqdm import tqdm

import torch
from functools import partial

from gm1_utils import GM1D, GM1DDenoiser
from samplers import bridge_statistics
from algorithms import cfg_denoiser

import matplotlib.pyplot as plt


# --- prior
prior = GM1D(
    means=torch.tensor([-2.0, 2.0]),
    covariances=torch.tensor([1.0, 1.0]),
    weights=torch.tensor([0.5, 0.5]),
)

# --- conditional
conditional = GM1D(
    means=torch.tensor([1.5, 2.5]),
    covariances=torch.tensor([0.1, 0.1]),
    weights=torch.tensor([0.5, 0.5]),
)


for denoiser_type in ["perfect", "cfg"]:

    for guidance in (3,):

        # arrows will be of the form {sigma: {x_t, renyi, cfg}}
        arrows = {}

        # sigmas where to show velocity field
        idx_sigmas = [3, 6]

        rho = 7.0
        sigma_min = 2e-2
        sigma_max = 100
        n_steps = 50

        n_samples = 1_000

        eta = 0.0
        x_0 = torch.linspace(-100, 100, 2000)

        gm_denoiser = GM1DDenoiser(prior, conditional)
        d_perfect_fn = partial(
            gm_denoiser.perfect_denoiser_fn,
            guidance=guidance,
            x_0=x_0,
            return_terms=True,
        )
        d_cfg_fn = cfg_denoiser(
            denoiser_fn=partial(gm_denoiser.denoiser_fn),
            cfg_cond_fn=lambda sigma, cfg_scale: cfg_scale > 1.0,
            cfg_scale=guidance,
        )

        selected_particles = torch.tensor([-1.5, -0.75, 0.0, 0.75, 1.5])
        sigmas = gm_denoiser.mk_sigmas_fn(n_steps, sigma_min, sigma_max, rho=rho)
        sigmas_flipped = sigmas.flip(0)
        pbar = tqdm(zip(sigmas_flipped[:-1], sigmas_flipped[1:]))

        selected_x_t = sigmas[-1] * selected_particles
        arr_intermediate = [selected_x_t]

        # used to plot historgram and fields
        x_t = sigmas[-1] * torch.randn((n_samples,))
        arr_x_t = [x_t]
        for sigma_t, sigma_t_prev in pbar:

            if sigma_t in sigmas[idx_sigmas]:
                selected_x_0t, renyi, cfg = d_perfect_fn(x=selected_x_t, sigma=sigma_t)
                idx = sigma_t.item()
                arrows[idx] = {}

                arrows[idx]["x_t"] = selected_x_t
                arrows[idx]["renyi"] = renyi
                arrows[idx]["cfg"] = cfg

            match denoiser_type:
                case "perfect":
                    selected_x_0t, *_ = d_perfect_fn(x=selected_x_t, sigma=sigma_t)
                    x_0t, *_ = d_perfect_fn(x=x_t, sigma=sigma_t)
                case "cfg":
                    selected_x_0t = d_cfg_fn(x=selected_x_t, sigma=sigma_t)
                    x_0t = d_cfg_fn(x=x_t, sigma=sigma_t)
                case _:
                    raise ValueError(
                        f"`denoiser_type` can only be 'cfg' or 'perfect', got {denoiser_type}"
                    )

            # update selected particles
            mean, std = bridge_statistics(
                selected_x_t, selected_x_0t, sigma_t, sigma_t_prev, eta
            )
            selected_x_t = mean + std * torch.randn_like(selected_x_t)
            arr_intermediate.append(selected_x_t)

            # update x_t and save intermediate
            mean, std = bridge_statistics(x_t, x_0t, sigma_t, sigma_t_prev, eta)
            x_t = mean + std * torch.randn_like(x_t)
            arr_x_t.append(x_t)

        # plotting
        i_min, i_max = 0, 30
        x_range = torch.linspace(-4.0, 8.0, 1000)
        sig_range = torch.linspace(sigmas[i_min], sigmas[i_max], 100)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(4.5, 2.5),
            gridspec_kw={
                "width_ratios": [1, 4],
                "wspace": 0.05,
            },
            sharey=True,
        )
        axes[0].invert_xaxis()

        # -- plot densities
        axes[0].plot(
            prior.log_prob(x_range).exp(),
            x_range,
            label=r"$p(x)$",
            color="blue",
            linestyle="dashed",
            linewidth=1.0,
        )

        axes[0].plot(
            conditional.log_prob(x_range).exp(),
            x_range,
            label=r"$p(x|c)$",
            color="blue",
            linewidth=1.0,
        )

        density_guided = (
            guidance * conditional.log_prob(x_range)
            + (1 - guidance) * prior.log_prob(x_range)
        ).exp()
        const_int = (density_guided[:-1] * x_range.diff()).sum()
        axes[0].plot(
            density_guided / const_int,
            x_range,
            label=r"$p(x|c)^w p(x)^{1-w}$" f" | $w$={guidance}",
            color="red",
            linewidth=1.0,
        )
        # --- plot historgram of particles
        axes[0].hist(
            x=x_t,
            density=True,
            alpha=0.7,
            bins=10,
            orientation="horizontal",
            color="lightgray",
            edgecolor="black",
            linewidth=0.5,
        )

        # --- plot the trajectories of the selected particles
        arr_intermediate = torch.stack(arr_intermediate).flip(0).cpu()
        axes[1].plot(
            sigmas[i_min:i_max],
            arr_intermediate[i_min:i_max],
            color="black",
            alpha=1.0,
            linewidth=0.4,
        )

        arr_x_t = torch.stack(arr_x_t).flip(0).cpu()
        axes[1].plot(
            sigmas[i_min:i_max],
            arr_x_t[i_min:i_max],
            color="red",
            alpha=0.03,
            linewidth=0.5,
        )

        # --- plot velocity of the selected particles
        s = 2.5
        config_quiver = {
            "angles": "xy",  # to plot arrows correctly as (x, y) (dx, dy)
            # for pretty arrows
            "width": s * 0.02,
            "headwidth": s * 2.0,
            "headlength": s * 2.5,
            "scale": 1.5,
            "units": "xy",
        }

        for sigma_t in sigmas[idx_sigmas]:
            sigma_t = sigma_t.item()
            Y = arrows[sigma_t]["x_t"]
            X = [torch.tensor(sigma_t)] * len(Y)
            # time derivative
            U = -torch.ones_like(Y)

            # velocity: sigma_t \nabla log p_t
            renyi_velocity = sigma_t * arrows[sigma_t]["renyi"]
            cfg_velocity = sigma_t * arrows[sigma_t]["cfg"]
            perfect_velociy = renyi_velocity + cfg_velocity

            q_cfg = axes[1].quiver(X, Y, U, cfg_velocity, color="red", **config_quiver)
            q_perfect = axes[1].quiver(
                X, Y, U, perfect_velociy, color="black", **config_quiver
            )
            # renyi contribution
            q_renyi = axes[1].quiver(
                X, Y, 0.0 * U, renyi_velocity, color="blue", **config_quiver
            )

        # ---

        axes[1].set_xlim(0, sigmas[i_max - 1])
        axes[0].set_ylim(x_range.min(), x_range.max())
        axes[0].set_xlim(2, 0)

        axes[0].set_xticks([2])
        axes[0].set_yticks([-3, 0, 3, 6])
        axes[1].set_xticks([0, 3, 6, 9])

        axes[0].tick_params(axis="both")
        axes[1].tick_params(axis="both")

        axes[0].set_ylabel(r"$\mathbf{x}$")
        # axes[0].set_xlabel("fraction")
        axes[1].set_xlabel(r"$\sigma$")

        fig.savefig(
            f"fig_{denoiser_type}_{guidance}.pdf",
            bbox_inches="tight",
        )
