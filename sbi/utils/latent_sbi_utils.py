import torch
from torch import nn, Tensor
from sbi.utils import standardizing_net
import torch_interpolations
import time
from math import prod


class EmbedTheta(nn.Module):
    r"""
    Network that takes in $\theta$ and $z$ and outputs the embedded $\theta$.

    $z$ is left unaltered.
    """

    def __init__(self, y_embedding, y_dim):
        super().__init__()
        self.net_y = y_embedding
        self.y_dim = y_dim

    def forward(self, y_z):
        y = y_z[:, : self.y_dim]
        z = y_z[:, self.y_dim :]
        embedded_y = self.net_y(y)
        embedded_yz = torch.cat([embedded_y, z], dim=1)
        return embedded_yz


class MergeNet(nn.Module):
    r"""
    Flow that models $p(x | y, z)$.

    This also implements the embedding of $z$: $\psi = f(z)$.
    """

    def __init__(
        self, net, embedding_z, y_dim, z_score_psi, batch_z=None, batch_theta=None
    ):
        super().__init__()
        self.flow_given_theta_psi = net
        if z_score_psi:

            class FeedForwardThetaZandZScore(nn.Module):
                """
                This is used instead of nn.Sequential because self.net_z takes two args.
                """

                def __init__(self, net, batch_z, batch_theta):
                    super().__init__()
                    self.net = net
                    psi = self.net(batch_z, batch_theta).detach()
                    self.standardize_psi = standardizing_net(psi)

                def forward(self, theta, z):
                    psi = self.net(theta, z)
                    norm_psi = self.standardize_psi(psi)
                    return norm_psi

            self.net_z = FeedForwardThetaZandZScore(embedding_z, batch_z, batch_theta)

        self.y_dim = y_dim

    def forward(self, context: Tensor):
        y = context[:, : self.y_dim]
        z = context[:, self.y_dim :]
        embedded_z = self.net_z(z, y)
        embedded_yz = torch.cat([y, embedded_z], dim=1)
        return embedded_yz

    def log_prob(self, x: Tensor, context: Tensor):
        embedded_yz = self.forward(context)
        return self.flow_given_theta_psi.log_prob(x, context=embedded_yz)

    def sample(self, num_samples: int, context):
        embedded_yz = self.forward(context)
        return self.flow_given_theta_psi.sample(num_samples, context=embedded_yz)


class PsiPrior(torch.distributions.MultivariateNormal):
    r"""
    Prior over $\psi$ given a prior over $z$ and an embedding net $\psi = f(z)$.
    """

    def __init__(
        self, prior_z, prior_theta, embedding_net_z, eval_method="kde_interpolate"
    ):
        dim = prior_z.sample((1,)).shape[1]
        super().__init__(torch.zeros(dim), torch.eye(dim))
        self.prior_z = prior_z
        self.prior_theta = prior_theta
        self.embedding_net_z = embedding_net_z
        if eval_method == "kde_interpolate":
            self.evaluator = KDEInterpolate(prior_psi=self)
        elif eval_method == "kde":
            self.evaluator = KDE(prior_psi=self)
        else:
            raise NotImplementedError

    def sample(self, sample_shape=(1,), max_sampling_batch_size: int = 100_000):
        num_to_draw = min(prod(sample_shape), max_sampling_batch_size)
        num_drawn = 0
        all_samples = []
        while num_drawn < prod(sample_shape):
            z_samples = self.prior_z.sample(sample_shape)
            theta_samples = self.prior_theta.sample(sample_shape)
            # `.detach()` to fix #5.
            psi = self.embedding_net_z(z_samples, theta_samples).detach()
            all_samples.append(psi)
            num_drawn += num_to_draw
        return torch.cat(all_samples)

    def log_prob(self, psi):
        log_prob = self.evaluator.log_prob(psi)
        return log_prob


class KDEInterpolate:
    """
    Linear interpolation for a KDE based on samples.
    """

    def __init__(self, prior_psi):
        self.rgi = None
        self.prior_psi = prior_psi

    def update_state(self):
        self.build_kde()

    def build_kde(
        self,
        num_samples: int = 100_000,
        num_bins: int = 100,
    ):
        psi_samples = self.prior_psi.sample((num_samples,))
        self.minimum, _ = torch.min(psi_samples, dim=0)
        self.maximum, _ = torch.max(psi_samples, dim=0)
        kde_vals = torch.histc(
            psi_samples, bins=num_bins, min=self.minimum.item(), max=self.maximum.item()
        ).detach()
        bin_width = (self.maximum.item() - self.minimum.item()) / num_bins
        hist_positions = torch.linspace(
            self.minimum.item() + bin_width / 2,
            self.maximum.item() - bin_width / 2,
            num_bins,
        )
        self.rgi = torch_interpolations.RegularGridInterpolator(
            (hist_positions,),
            kde_vals,
        )

    def eval_kde(self, psi: Tensor):
        values_under_kde = self.rgi((psi,))
        values_under_kde[psi < self.minimum] = 0.0
        values_under_kde[psi > self.maximum] = 0.0
        return values_under_kde / 1000.0

    def log_prob(self, psi):
        if self.rgi is None:
            self.build_kde()

        return torch.log(self.eval_kde(psi))


class KDE:
    """
    KDE where the returned log-prob is based on the nearest neighbour.
    """

    def __init__(self, prior_psi):
        self.kde = None
        self.prior_psi = prior_psi

    def update_state(self):
        self.build_kde()

    def build_kde(
        self,
        num_samples: int = 1_000_000,
        num_bins: int = 100,
    ):
        psi_samples = self.prior_psi.sample((num_samples,))

        self.minimum, _ = torch.min(psi_samples, dim=0)
        self.maximum, _ = torch.max(psi_samples, dim=0)
        self.kde = torch.histc(
            psi_samples, bins=num_bins, min=self.minimum.item(), max=self.maximum.item()
        ).detach()

    def eval_kde(self, psi: Tensor):
        num_bins = self.kde.shape[0]
        psi_minus_min = psi - self.minimum
        fraction_within_range = psi_minus_min / (self.maximum - self.minimum)
        selected_bins = fraction_within_range * num_bins
        selected_bins[selected_bins < 0] = 0
        selected_bins[selected_bins > num_bins - 1] = num_bins - 1
        inds = torch.as_tensor(selected_bins, dtype=torch.long)
        kde_values = self.kde[inds]
        kde_values[psi < self.minimum] = 0.0
        kde_values[psi > self.maximum] = 0.0
        return kde_values / 1000.0

    def log_prob(self, psi):
        if self.kde is None:
            self.build_kde()

        return torch.log(self.eval_kde(psi))
