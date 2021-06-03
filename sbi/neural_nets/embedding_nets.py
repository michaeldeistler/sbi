import torch
from torch import nn, Tensor
from sbi.utils.sbiutils import standardizing_net


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
        self,
        net,
        embedding_z,
        y_dim,
        z_score_z,
        z_score_theta_for_z,
        z_score_psi,
        batch_theta=None,
        batch_z=None,
    ):
        super().__init__()

        self.flow_given_theta_psi = net

        class FeedForwardThetaZandZScore(nn.Module):
            """
            This is used instead of nn.Sequential because self.net_z takes two args.
            """

            def __init__(
                self,
                net,
                z_score_z,
                z_score_theta_for_z,
                z_score_psi,
                batch_theta,
                batch_z,
            ):
                super().__init__()
                self.net = net
                self.standardize_z = maybe_z_score_net(z_score_z, batch_z)
                self.standardize_theta_for_z = maybe_z_score_net(
                    z_score_theta_for_z, batch_theta
                )
                psi = self.forward_no_z_score_psi(batch_theta, batch_z).detach()
                self.standardize_psi = maybe_z_score_net(z_score_psi, psi)

            def forward_no_z_score_psi(self, theta, z):
                z = self.standardize_z(z)
                theta = self.standardize_theta_for_z(theta)
                psi = self.net(theta, z)
                return psi

            def forward(self, theta, z):
                psi = self.forward_no_z_score_psi(theta, z)
                norm_psi = self.standardize_psi(psi)
                return norm_psi

        self.net_z = FeedForwardThetaZandZScore(
            embedding_z,
            z_score_z,
            z_score_theta_for_z,
            z_score_psi,
            batch_theta,
            batch_z,
        )

        self.y_dim = y_dim

    def forward(self, context: Tensor):
        y = context[:, : self.y_dim]
        z = context[:, self.y_dim :]
        embedded_z = self.net_z(y, z)
        embedded_yz = torch.cat([y, embedded_z], dim=1)
        return embedded_yz

    def log_prob(self, x: Tensor, context: Tensor):
        embedded_yz = self.forward(context)
        return self.flow_given_theta_psi.log_prob(x, context=embedded_yz)

    def sample(self, num_samples: int, context):
        embedded_yz = self.forward(context)
        return self.flow_given_theta_psi.sample(num_samples, context=embedded_yz)


def maybe_z_score_net(z_score, data_batch):
    if z_score:
        standardization_net = standardizing_net(data_batch)
    else:
        standardization_net = nn.Identity()
    return standardization_net
