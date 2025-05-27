import torch
from torch import nn
from torch import Tensor
from torch.distributions import MixtureSameFamily, Categorical, Normal

from torch.func import grad


class GM1D(MixtureSameFamily):
    def __init__(self, means, covariances, weights):
        mixture_distribution = Categorical(probs=weights)
        component_distribution = Normal(
            loc=means,
            scale=covariances.sqrt(),
        )

        super().__init__(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        self.means = means
        self.weights = weights
        self.covariances = covariances
        self.log_weights = weights.log()


class GM1DDenoiser(nn.Module):

    def __init__(
        self,
        prior: GM1D,
        posterior: GM1D,
    ):
        super().__init__()
        self.prior = prior
        self.posterior = posterior

    def denoiser_fn(self, x, sigma, use_cfg: bool = False, **kwargs):
        cfg_scale = kwargs.get("cfg_scale", 1.0)

        conv_posterior = conv_mixture(self.posterior, sigma)
        posterior_score = grad(lambda z: conv_posterior.log_prob(z).sum())(x)

        if use_cfg:
            conv_prior = conv_mixture(self.prior, sigma)
            prior_score = grad(lambda z: conv_prior.log_prob(z).sum())(x)
            return torch.cat([x] * 2) + sigma**2.0 * torch.cat(
                [prior_score, posterior_score], dim=0
            )
        else:
            return x + sigma**2.0 * cfg_scale * posterior_score

    def perfect_denoiser_fn(
        self, x, sigma, guidance, x_0, return_terms=False, **kwargs
    ):

        cfg_term = self.cfg_score(x, sigma, guidance)
        renyi_term = self.cfg_additional_term(x, sigma, guidance, x_0)

        if return_terms:
            return x + sigma**2 * (renyi_term + cfg_term), renyi_term, cfg_term

        return x + sigma**2 * (renyi_term + cfg_term)

    def mk_sigmas_fn(
        self,
        n_steps: int,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
    ) -> Tensor:
        sigmas = (
            sigma_max ** (1 / rho)
            + torch.linspace(1, 0, n_steps)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        return sigmas

    def cfg_score(self, x, sigma, cfg_guidance):

        fn = grad(lambda input: self.cfg_log_density(input, sigma, cfg_guidance).sum())

        return fn(x)

    def cfg_additional_term(self, x, sigma, cfg_guidance, x_0: Tensor):

        fn = grad(
            lambda input: self.cfg_log_renyi(input, sigma, cfg_guidance, x_0).sum()
        )

        return fn(x)

    def cfg_log_renyi(self, x, sigma, cfg_guidance, x_0: Tensor):
        # compute log int post_0t^w  prior_0t^{1-w} dx_0
        # NOTE x_0 must be a tensor that doesn't require grad

        # skip case where there w == 1 (guidance)
        if cfg_guidance == 1.0:
            return torch.zeros_like(x, requires_grad=x.requires_grad)

        # compute log [post_0t ^w  prior_0t ^{1-w}]
        log_prior_0t = GM1DDenoiser.vectorized_log_dist_0t(x, sigma, x_0, self.prior)
        log_posterior_0t = GM1DDenoiser.vectorized_log_dist_0t(
            x, sigma, x_0, self.posterior
        )
        log_p_powered = (
            cfg_guidance * log_posterior_0t + (1 - cfg_guidance) * log_prior_0t
        )

        # compute the integral using trapezoid method
        log_dx_0 = x_0.diff().log()
        log_2 = torch.tensor(2.0).log()
        part_1 = torch.logsumexp(log_p_powered[:, 1:] + log_dx_0 - log_2, dim=-1)
        part_2 = torch.logsumexp(log_p_powered[:, :-1] + log_dx_0 - log_2, dim=-1)

        return torch.logsumexp(torch.stack((part_1, part_2)), dim=0)

    def cfg_log_density(self, x, sigma, cfg_guidance):

        prior_log = GM1DDenoiser.batched_log_p_t(x, sigma, self.prior)
        posterior_log = GM1DDenoiser.batched_log_p_t(x, sigma, self.posterior)

        # efficient way to compute cfg_guidance * posterior_score + (1 - cfg_guidance) * prior_score
        return torch.lerp(prior_log, posterior_log, cfg_guidance)

    @staticmethod
    def batched_log_p_t(x, sigma, mixture: GM1D):
        # sigma is used as a scaler
        means = mixture.means
        covs = mixture.covariances + sigma**2
        log_weights = mixture.log_weights

        log_prob_prior = torch.logsumexp(
            Normal(loc=means, scale=covs.sqrt()).log_prob(x[:, None]) + log_weights,
            dim=-1,
        )

        return log_prob_prior

    @staticmethod
    def vectorized_log_fwd_t0(x_t, sigma, x_0):
        # NOTE vectorized with respect to x_t, x_0; sigma is a scaler
        return Normal(loc=x_0, scale=sigma).log_prob(x_t[:, None])

    @staticmethod
    def vectorized_log_dist_0t(x_t, sigma, x_0, dist: GM1D):
        # NOTE vectorized with respect to x_t, x_0; sigma is a scaler
        log_prior = dist.log_prob(x_0)
        log_fwd = GM1DDenoiser.vectorized_log_fwd_t0(x_t, sigma, x_0)
        log_p_t = GM1DDenoiser.batched_log_p_t(x_t, sigma, dist)

        log_p_0t = log_prior[None] + log_fwd - log_p_t[:, None]
        return log_p_0t

    # NOTE: used only for debugging purposes
    def _log_renyi_one_Gaussian(self, x, sigma, cfg_guidance):
        # this valid only if prior and posterior have one mode
        prior_0t = _dist_p_0t(x, sigma, self.prior.component_distribution)
        posterior_0t = _dist_p_0t(x, sigma, self.posterior.component_distribution)

        return _analytic_log_renyi(prior_0t, posterior_0t, cfg_guidance)


def conv_mixture(
    mixture: GM1D,
    sigma: Tensor,
):
    n_mixtures = mixture.means.shape[0]
    inflated_covs = mixture.covariances + sigma**2.0 * torch.ones(n_mixtures)

    return GM1D(means=mixture.means, covariances=inflated_covs, weights=mixture.weights)


def linear_conditioning(
    mixture: GM1D,
    obs: Tensor,
    operator: Tensor,
    std_obs: Tensor,
) -> GM1D:

    cond_precision_matrix = 1 / std_obs**2
    cond_covariance_matrix = std_obs**2

    mean = mixture.component_distribution.loc
    covariance = mixture.component_distribution.scale**2
    log_weights = mixture.mixture_distribution.logits

    new_covariances = 1 / (1 / covariance + operator**2 * cond_precision_matrix)
    new_means = new_covariances * (
        operator * cond_precision_matrix * obs + mean / covariance
    )

    d_y = Normal(
        loc=operator * mean,
        scale=(operator**2 * covariance + cond_covariance_matrix).sqrt(),
    )
    new_log_weights = log_weights + d_y.log_prob(obs)
    new_weights = new_log_weights.softmax(0).flatten()

    return GM1D(new_means, new_covariances, new_weights)


# ---
# NOTE: these are used for debugging purposes
def _analytic_log_renyi(dist_0: Normal, dist_1: Normal, cfg_guidance):
    # compute log int dist_1 ^w  dist_0 ^{1-w} dx_0
    # when dist_0 and dist_1 are 1D Gaussian
    cov_0 = dist_0.scale**2
    cov_1 = dist_1.scale**2
    cov_alpha = cfg_guidance * cov_0 + (1 - cfg_guidance) * cov_1

    # cov part
    cov_term = (
        0.5 * (cov_0**cfg_guidance * cov_1 ** (1 - cfg_guidance) / cov_alpha).log()
    )
    # mean part
    mean_0 = dist_0.mean
    mean_1 = dist_1.mean
    mean_term = (
        0.5 * (cfg_guidance * (cfg_guidance - 1) / cov_alpha) * (mean_1 - mean_0) ** 2
    )

    return cov_term + mean_term


def _dist_p_0t(x_t, sigma, dist: Normal):
    # distribution p_0|t when applying diffusion on a 1D Gaussian
    # sigma must be a scaler
    mean = dist.mean
    cov = dist.scale**2

    cov_0t = (cov * sigma**2) / (cov + sigma**2)
    mean_0t = cov_0t * (mean / cov + x_t / sigma**2)

    return Normal(loc=mean_0t, scale=cov_0t.sqrt())
