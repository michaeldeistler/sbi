import torch
from torch import nn
import matplotlib.pyplot as plt


class BellShapedFeatures(nn.Module):
    def __init__(self, dim_z: int, dim_psi: int):
        super().__init__()

        init_means = torch.linspace(0, 1, dim_psi + 2)[1:-1].unsqueeze(1)
        self._means_of_weights = nn.Parameter(init_means)

        init_std = max(0.5 / dim_psi, 0.2)
        self._stds_of_weights = nn.Parameter(
            torch.tensor([init_std] * dim_psi).unsqueeze(1)
        )

        init_height = 1.0
        self._heights_of_weights = nn.Parameter(
            torch.tensor([init_height] * dim_psi).unsqueeze(1)
        )
        self.dim_z = dim_z
        self.dim_psi = dim_psi
        self.prior_independent_of_theta = True

    def forward(self, theta, z):
        stds = torch.exp(self._stds_of_weights)
        heights = self._heights_of_weights / torch.sum(
            torch.abs(self._heights_of_weights)
        )
        positions = torch.linspace(0, 1, self.dim_z).unsqueeze(0)
        positions = positions.repeat(self.dim_psi, 1)
        weights = (
            heights
            / stds
            * torch.exp(-((positions - self._means_of_weights) ** 2) * 0.5 / stds ** 2)
        )
        repeated_z = z.unsqueeze(1)
        repeated_z = repeated_z.repeat(1, self.dim_psi, 1)
        psi = torch.mean(weights * repeated_z, dim=2)

        return psi


class MeanNet(nn.Module):
    def __init__(
        self,
        dim_z: int,
        dim_psi: int,
        dim_theta: int,
        num_gaussians: int = 10,
        num_hiddens: int = 10,
        encoding_net: str = "linear",
    ):
        super().__init__()

        self.bell_features = BellShapedFeatures(dim_z, num_gaussians)
        if encoding_net == "identity":
            assert num_gaussians == dim_psi
            self.encoding_net = nn.Identity()
        elif encoding_net == "linear":
            self.encoding_net = nn.Linear(num_gaussians, dim_psi)
        elif encoding_net == "mlp":
            self.encoding_net = nn.Sequential(
                nn.Linear(num_gaussians, num_hiddens),
                nn.BatchNorm1d(num_hiddens),
                nn.ReLU(),
                nn.Linear(num_hiddens, num_hiddens),
                nn.BatchNorm1d(num_hiddens),
                nn.ReLU(),
                nn.Linear(num_hiddens, num_hiddens),
                nn.BatchNorm1d(num_hiddens),
                nn.ReLU(),
                nn.Linear(num_hiddens, dim_psi),
            )
        else:
            raise NameError

        self.dim_z = dim_z
        self.dim_psi = dim_psi
        self.prior_independent_of_theta = True

    def forward(self, theta, z):
        features = self.bell_features(theta, z)
        psi = self.encoding_net(features)

        return psi


class BellShapedFeaturesGivenTheta(nn.Module):
    def __init__(self, dim_z: int, dim_psi: int, dim_theta: int, num_hiddens: int = 10):
        super().__init__()

        self.net_to_predict_means_and_stds = nn.Sequential(
            nn.Linear(dim_theta, num_hiddens),
            nn.BatchNorm1d(num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, num_hiddens),
            nn.BatchNorm1d(num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, num_hiddens),
            nn.BatchNorm1d(num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, 3 * dim_psi),
        )
        # (3 * ) because means, stds, heights

        self.dim_z = dim_z
        self.dim_psi = dim_psi
        self.prior_independent_of_theta = False

    def forward(self, theta, z):
        means_stds_heights = self.net_to_predict_means_and_stds(theta)
        means = means_stds_heights[:, : 1 * self.dim_psi]
        stds = torch.exp(means_stds_heights[:, 1 * self.dim_psi : 2 * self.dim_psi])
        heights = means_stds_heights[:, 2 * self.dim_psi :]
        heights = heights / torch.sum(torch.abs(heights), dim=1).unsqueeze(1)
        means = means.unsqueeze(2).repeat(1, 1, self.dim_z)
        stds = stds.unsqueeze(2).repeat(1, 1, self.dim_z)
        heights = heights.unsqueeze(2).repeat(1, 1, self.dim_z)
        positions = torch.linspace(0, 1, self.dim_z).unsqueeze(0).unsqueeze(0)
        positions = positions.repeat(theta.shape[0], self.dim_psi, 1)
        weights = (
            heights / stds * torch.exp(-((positions - means) ** 2) * 0.5 / stds ** 2)
        )
        repeated_z = z.unsqueeze(1)
        repeated_z = repeated_z.repeat(1, self.dim_psi, 1)
        psi = torch.mean(weights * repeated_z, dim=2)

        return psi


class MeanNetGivenTheta(nn.Module):
    def __init__(
        self,
        dim_z: int,
        dim_psi: int,
        dim_theta: int,
        num_gaussians: int = 10,
        num_hiddens_mean_net: int = 10,
        num_hiddens_encoding_net: int = 10,
        encoding_net: str = "linear",
    ):
        super().__init__()

        self.bell_features = BellShapedFeaturesGivenTheta(
            dim_z=dim_z,
            dim_psi=num_gaussians,
            dim_theta=dim_theta,
            num_hiddens=num_hiddens_mean_net,
        )
        if encoding_net == "identity":
            assert num_gaussians == dim_psi
            self.encoding_net == nn.Identity()
        elif encoding_net == "linear":
            self.encoding_net = nn.Linear(num_gaussians + dim_theta, dim_psi)
        elif encoding_net == "mlp":
            self.encoding_net = nn.Sequential(
                nn.Linear(num_gaussians + dim_theta, num_hiddens_encoding_net),
                nn.BatchNorm1d(num_hiddens_encoding_net),
                nn.ReLU(),
                nn.Linear(num_hiddens_encoding_net, num_hiddens_encoding_net),
                nn.BatchNorm1d(num_hiddens_encoding_net),
                nn.ReLU(),
                nn.Linear(num_hiddens_encoding_net, num_hiddens_encoding_net),
                nn.BatchNorm1d(num_hiddens_encoding_net),
                nn.ReLU(),
                nn.Linear(num_hiddens_encoding_net, dim_psi),
            )
        else:
            raise NameError

        self.dim_z = dim_z
        self.dim_psi = dim_psi
        self.prior_independent_of_theta = False

    def forward(self, theta, z):
        mean_embedding = self.bell_features(theta, z)
        psi = self.encoding_net(torch.cat((mean_embedding, theta), dim=1))

        return psi


class MeanNetDemo(nn.Module):
    """
    Used only to make a figure...
    """

    def __init__(self, dim_z: int, dim_psi: int):
        super().__init__()

        init_means = torch.linspace(0, 1, dim_psi + 2)[1:-1].unsqueeze(1)
        self._means_of_weights = nn.Parameter(init_means)

        init_std = max(0.5 / dim_psi, 0.2) / 2.0
        self._stds_of_weights = nn.Parameter(
            torch.tensor([init_std] * dim_psi).unsqueeze(1)
        )

        init_height = 1.0
        self._heights_of_weights = nn.Parameter(
            torch.tensor([init_height] * dim_psi).unsqueeze(1)
        )
        self.dim_z = dim_z
        self.dim_psi = dim_psi
        self.prior_independent_of_theta = True

    def forward(self, theta, z):
        positions = torch.linspace(0, 1, self.dim_z).unsqueeze(0)
        positions = positions.repeat(self.dim_psi, 1)
        weights = self._heights_of_weights * torch.exp(
            -((positions - self._means_of_weights) ** 2)
            * 0.5
            / self._stds_of_weights ** 2
        )
        repeated_z = z.unsqueeze(1)
        repeated_z = repeated_z.repeat(1, self.dim_psi, 1)
        psi = torch.mean(weights * repeated_z, dim=2)

        return psi, weights