# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from pyknos.mdn.mdn import MultivariateGaussianMDN
from pyknos.nflows import flows, transforms
from torch import Tensor, nn

import sbi.utils as utils
from sbi.neural_nets.embedding_nets import MergeNet, EmbedTheta


def build_mdn(
    batch_x: Tensor = None,
    batch_y: Tensor = None,
    z_score_x: bool = True,
    z_score_y: bool = True,
    hidden_features: int = 50,
    num_components: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    **kwargs,
) -> nn.Module:
    """Builds MDN p(x|y).

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network.
        z_score_y: Whether to z-score ys passing into the network.
        hidden_features: Number of hidden features.
        num_components: Number of components.
        embedding_net: Optional embedding network for y.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for MDNs and are therefore ignored.

    Returns:
        Neural network.
    """
    x_numel = batch_x[0].numel()
    # Infer the output dimensionality of the embedding_net by making a forward pass.
    y_numel = embedding_net(batch_y[:1]).numel()

    transform = transforms.IdentityTransform()

    if z_score_x:
        transform_zx = utils.standardizing_transform(batch_x)
        transform = transforms.CompositeTransform([transform_zx, transform])

    if z_score_y:
        embedding_net = nn.Sequential(utils.standardizing_net(batch_y), embedding_net)

    distribution = MultivariateGaussianMDN(
        features=x_numel,
        context_features=y_numel,
        hidden_features=hidden_features,
        hidden_net=nn.Sequential(
            nn.Linear(y_numel, hidden_features),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        ),
        num_components=num_components,
        custom_initialization=True,
    )

    neural_net = flows.Flow(transform, distribution, embedding_net)

    return neural_net


def build_latent_mdn(
    batch_x: Tensor = None,
    batch_y: Tensor = None,
    batch_z: Tensor = None,
    z_score_x: bool = True,
    z_score_y: bool = True,
    z_score_z: bool = False,
    z_score_psi: bool = True,
    hidden_features: int = 50,
    num_components: int = 10,
    embedding_net_y: nn.Module = nn.Identity(),
    embedding_net_z: nn.Module = nn.Identity(),
    embedding_net_y_psi: nn.Module = nn.Identity(),
    **kwargs,
) -> nn.Module:
    """Builds MDN p(x|y).

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network.
        z_score_y: Whether to z-score ys passing into the network.
        hidden_features: Number of hidden features.
        num_components: Number of components.
        embedding_net: Optional embedding network for y.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for MDNs and are therefore ignored.

    Returns:
        Neural network.
    """
    x_numel = batch_x[0].numel()
    # Infer the output dimensionality of the embedding_net by making a forward pass.
    y_numel = embedding_net_y(batch_y[:1]).numel()
    psi_numel = embedding_net_z(batch_z[:1]).numel()

    transform = transforms.IdentityTransform()

    if z_score_x:
        transform_zx = utils.standardizing_transform(batch_x)
        transform = transforms.CompositeTransform([transform_zx, transform])

    if z_score_y:
        embedding_net_y = nn.Sequential(
            utils.standardizing_net(batch_y), embedding_net_y
        )

    print("num_components", num_components)

    distribution = MultivariateGaussianMDN(
        features=x_numel,
        context_features=y_numel + psi_numel,
        hidden_features=hidden_features,
        hidden_net=nn.Sequential(
            nn.Linear(y_numel + psi_numel, hidden_features),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        ),
        num_components=num_components,
        custom_initialization=True,
    )

    if z_score_x:
        transform_zx = utils.standardizing_transform(batch_x)
        transform = transforms.CompositeTransform([transform_zx, transform])

    if z_score_y:
        embedding_net_y = nn.Sequential(
            utils.standardizing_net(batch_y), embedding_net_y
        )
    if z_score_z:
        embedding_net_z = nn.Sequential(
            utils.standardizing_net(batch_z), embedding_net_z
        )

    net_that_encodes_only_y = EmbedTheta(embedding_net_y, batch_y.shape[1])
    embedding_net_y_psi = nn.Sequential(
        net_that_encodes_only_y,
        embedding_net_y_psi,
    )

    neural_net = flows.Flow(transform, distribution, embedding_net_y_psi)

    wrapped_net = MergeNet(
        net=neural_net,
        embedding_z=embedding_net_z,
        y_dim=batch_y.shape[1],
        z_score_psi=z_score_psi,
        batch_z=batch_z,
    )

    return wrapped_net
