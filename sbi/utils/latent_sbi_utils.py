import torch
from typing import Optional, Dict
from torch import nn, Tensor, optim
from torch.nn.functional import poisson_nll_loss
from sbi.utils.sbiutils import standardizing_net
import torch_interpolations
import time
from math import prod
import numpy as np
from sbi.utils.get_nn_models import likelihood_nn
from torch.utils import data
from torch.nn.utils import clip_grad_norm_
from sbi.utils.torchutils import ensure_theta_batched, atleast_2d_float32_tensor
from copy import deepcopy


# TODO is inheritance necessary?
class ThetaPsiPrior(torch.distributions.MultivariateNormal):
    def __init__(self, prior_theta, prior_psi):

        one_theta = prior_theta.sample((1,))
        dim = one_theta.shape[1] + prior_psi.sample((1,), one_theta).shape[1]
        super().__init__(torch.zeros(dim), torch.eye(dim))
        self.prior_theta = prior_theta
        self.prior_psi = prior_psi
        self.theta_dim = prior_theta.sample((1,)).shape[1]

    def sample(self, sample_shape=(1,)):
        theta = self.prior_theta.sample(sample_shape)
        psi = self.prior_psi.sample(sample_shape, theta=theta)
        return torch.cat((theta, psi), dim=1)

    def log_prob(self, theta_psi: Tensor):
        theta_psi = atleast_2d_float32_tensor(theta_psi)
        theta = theta_psi[:, : self.theta_dim]
        psi = theta_psi[:, self.theta_dim :]
        term1 = self.prior_theta.log_prob(theta)
        term2 = self.prior_psi.log_prob(psi=psi, theta=theta)
        return term1 + term2


# TODO is inheritance necessary?
class PsiPrior(torch.distributions.MultivariateNormal):
    r"""
    Prior over $\psi$ given a prior over $z$ and an embedding net $\psi = f(z)$.
    """

    def __init__(
        self,
        prior_z,
        prior_theta,
        embedding_net_z,
        eval_method="kde_interpolate",
        evaluator_kwargs: Dict = {},
    ):
        dim = prior_z.sample((1,)).shape[1]
        super().__init__(torch.zeros(dim), torch.eye(dim))
        self.prior_z = prior_z
        self.prior_theta = prior_theta
        self.embedding_net_z = embedding_net_z
        if eval_method == "kde_interpolate":
            check_if_prior_is_independent_of_theta(
                self.embedding_net_z.net, eval_method
            )
            self.evaluator = KDEInterpolate(prior_psi=self, prior_theta=prior_theta)
        elif eval_method == "kde":
            check_if_prior_is_independent_of_theta(
                self.embedding_net_z.net, eval_method
            )
            self.evaluator = KDE(prior_psi=self, prior_theta=prior_theta)
        elif eval_method == "kde_interpolate_np":
            check_if_prior_is_independent_of_theta(
                self.embedding_net_z.net, eval_method
            )
            self.evaluator = KDEInterpolateNP(prior_psi=self, prior_theta=prior_theta)
        elif eval_method == "flow":
            raise NotImplementedError
            # check_if_prior_is_independent_of_theta(self.embedding_net_z, eval_method)
            # self.evaluator = FlowAsPsiPrior(prior_psi=self)
        elif eval_method == "cond_flow":
            self.evaluator = CondFlowAsPsiPrior(
                prior_theta=prior_theta,
                prior_z=prior_z,
                embedding_net_z=embedding_net_z,
            )
        else:
            raise NameError

        # Build histogram or train flow.
        self.evaluator.update_state(**evaluator_kwargs)

    def sample(
        self,
        sample_shape=(1,),
        theta: Tensor = None,
        max_sampling_batch_size: int = 100_000,
    ):
        if theta is not None:
            assert theta.shape[0] == prod(sample_shape)
        num_to_draw = min(prod(sample_shape), max_sampling_batch_size)
        num_drawn = 0
        all_samples = []
        while num_drawn < prod(sample_shape):
            z_samples = self.prior_z.sample(sample_shape)
            # `.detach()` to fix #5.
            psi = self.embedding_net_z(theta, z_samples).detach()
            all_samples.append(psi)
            num_drawn += num_to_draw
        return torch.cat(all_samples)

    def log_prob(self, psi: Tensor, theta: Optional[Tensor] = None):
        log_prob = self.evaluator.log_prob(psi, theta)
        if log_prob.ndim > 1:
            log_prob = log_prob[:, 0]
        assert log_prob.shape[0] == psi.shape[0]
        assert log_prob.ndim == 1
        return log_prob


def check_if_prior_is_independent_of_theta(net, eval_method):
    if hasattr(net, "prior_independent_of_theta"):
        if net.prior_independent_of_theta:
            pass
        else:
            raise ValueError(
                f"You used an `embedding_net_z` that with "
                f"`prior_independent_of_theta=False`, but your "
                f"`eval_method={eval_method} which does not all that. Please use "
                f"`eval_method='flow'."
            )
    else:
        raise ValueError(
            f"Your `embedding_net_z` does not have an attribute "
            f"`prior_independent_of_theta`, but your "
            f"`eval_method={eval_method}. If your `embedding_net_z` is independent "
            f"of theta, please set `self.prior_independent_of_theta=True`"
        )


class CondFlowAsPsiPrior(nn.Module):
    def __init__(self, prior_theta, prior_z, embedding_net_z):
        super().__init__()
        self.prior_theta = prior_theta
        self.prior_z = prior_z
        self.net_z = embedding_net_z
        self.neural_net = None

    def update_state(self, **kwargs):
        self.train(**kwargs)

    def train(
        self,
        num_theta: int = 1_000,
        num_z_per_theta: int = 1,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        max_num_epochs: Optional[int] = None,
        stop_after_epochs: int = 20,
        clip_max_norm: Optional[float] = 5.0,
        num_transforms: int = 5,
        num_bins: int = 20,
        hidden_features: int = 50,
        **kwargs,
    ):
        theta = self.prior_theta.sample((num_theta,))
        theta = theta.repeat(num_z_per_theta, 1).detach()
        z = self.prior_z.sample((num_theta * num_z_per_theta,))
        psi = self.net_z(theta, z).detach()

        self.density_estimator_creator = likelihood_nn(
            "nsf",
            num_transforms=num_transforms,
            num_bins=num_bins,
            hidden_features=hidden_features,
        )
        self.neural_net = self.density_estimator_creator(theta, psi)

        dataset = data.TensorDataset(theta, psi)

        # Get total number of training examples.
        num_examples = len(dataset)

        # Select random train and validation splits from (theta, x) pairs.
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples

        permuted_indices = torch.randperm(num_examples)
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )
        train_loader = data.DataLoader(
            dataset,
            batch_size=min(training_batch_size, num_training_examples),
            drop_last=True,
            sampler=data.sampler.SubsetRandomSampler(train_indices),
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=min(training_batch_size, num_validation_examples),
            shuffle=False,
            drop_last=True,
            sampler=data.sampler.SubsetRandomSampler(val_indices),
        )

        optimizer = optim.Adam(
            list(self.neural_net.parameters()),
            lr=learning_rate,
        )
        self.epoch, self._val_log_prob = 0, float("-Inf")
        max_num_epochs = 2 ** 31 - 1 if max_num_epochs is None else max_num_epochs

        while self.epoch <= max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):

            # Train for a single epoch.
            self.neural_net.train()
            for batch in train_loader:
                optimizer.zero_grad()
                theta_batch, x_batch = (
                    batch[0],
                    batch[1],
                )
                # Evaluate on x with theta as context.
                log_prob = self.neural_net.log_prob(x_batch, context=theta_batch)
                loss = -torch.mean(log_prob)
                loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self.neural_net.parameters(),
                        max_norm=clip_max_norm,
                    )
                optimizer.step()

            self.epoch += 1

            # Calculate validation performance.
            self.neural_net.eval()
            log_prob_sum = 0
            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch = (
                        batch[0],
                        batch[1],
                    )
                    # Evaluate on x with theta as context.
                    log_prob = self.neural_net.log_prob(x_batch, context=theta_batch)
                    log_prob_sum += log_prob.sum().item()
            # Take mean over all validation samples.
            self._val_log_prob = log_prob_sum / (
                len(val_loader) * val_loader.batch_size
            )

            print(
                "Training neural network. Epochs trained: ",
                self.epoch,
                " / Validation log-prob: ",
                self._val_log_prob,
                end="\r",
            )

    def log_prob(self, psi: Tensor, theta: Tensor):
        if self.neural_net is None:
            self.update_state()

        psi = ensure_theta_batched(torch.as_tensor(psi))
        theta = atleast_2d_float32_tensor(theta)

        assert psi.shape[0] == theta.shape[0]

        return self.neural_net.log_prob(psi, context=theta)

    def _converged(self, epoch: int, stop_after_epochs: int) -> bool:
        """Return whether the training converged yet and save best model state so far.

        Checks for improvement in validation performance over previous epochs.

        Args:
            epoch: Current epoch in training.
            stop_after_epochs: How many fruitless epochs to let pass before stopping.

        Returns:
            Whether the training has stopped improving, i.e. has converged.
        """
        converged = False

        neural_net = self.neural_net

        # (Re)-start the epoch count with the first epoch or any improvement.
        if epoch == 0 or self._val_log_prob > self._best_val_log_prob:
            self._best_val_log_prob = self._val_log_prob
            self._epochs_since_last_improvement = 0
            self._best_model_state_dict = deepcopy(neural_net.state_dict())
        else:
            self._epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if self._epochs_since_last_improvement > stop_after_epochs - 1:
            neural_net.load_state_dict(self._best_model_state_dict)
            converged = True

        return converged


class KDEInterpolate:
    """
    Linear interpolation for a KDE based on samples.
    """

    def __init__(self, prior_psi, prior_theta):
        self.rgi = None
        self.prior_psi = prior_psi
        self.prior_theta = prior_theta

    def update_state(self, **kwargs):
        self.build_kde(**kwargs)

    def build_kde(self, num_samples: int = 100_000, num_bins: int = 100, **kwargs):
        theta_samples = self.prior_theta.sample((num_samples,))
        psi_samples = self.prior_psi.sample((num_samples,), theta=theta_samples)
        self.minimum, _ = torch.min(psi_samples, dim=0)
        self.maximum, _ = torch.max(psi_samples, dim=0)
        kde_vals = torch.histc(
            psi_samples,
            bins=num_bins,
            min=self.minimum.item(),
            max=self.maximum.item(),
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
        psi = psi.contiguous()
        values_under_kde = self.rgi((psi,))
        values_under_kde[psi < self.minimum] = 0.0
        values_under_kde[psi > self.maximum] = 0.0
        values_under_kde[values_under_kde == 0.0] = 1e-16
        return values_under_kde / 1000.0

    def log_prob(self, psi: Tensor, theta: Tensor):
        if self.rgi is None:
            self.build_kde()

        return torch.log(self.eval_kde(psi))


class KDEInterpolateNP:
    r"""
    Linear interpolation for a KDE based on samples.

    This uses `histogramdd` and can thus be used also for multi-dimensional $\psi$.
    """

    def __init__(self, prior_psi, prior_theta):
        self.rgi = None
        self.prior_psi = prior_psi
        self.prior_theta = prior_theta

    def update_state(self, **kwargs):
        self.build_kde(**kwargs)

    def build_kde(self, num_samples: int = 100_000, num_bins: int = 100, **kwargs):
        theta_samples = self.prior_theta.sample((num_samples,))
        psi_samples = self.prior_psi.sample((num_samples,), theta=theta_samples)
        self.minimum, _ = torch.min(psi_samples, dim=0)
        self.maximum, _ = torch.max(psi_samples, dim=0)
        maxima_minima = torch.stack([self.minimum, self.maximum]).T
        kde_vals, positions = np.histogramdd(
            psi_samples.numpy(), bins=num_bins, range=maxima_minima.tolist()
        )
        midpoints = [pos[:-1] + np.diff(pos) / 2 for pos in positions]
        hist_positions = [
            torch.as_tensor(pos, dtype=torch.float32) for pos in midpoints
        ]
        values = torch.as_tensor(kde_vals, dtype=torch.float32)
        self.rgi = torch_interpolations.RegularGridInterpolator(
            hist_positions,
            values,
        )

    def eval_kde(self, psi: Tensor):
        psi = psi.contiguous()
        values_under_kde = self.rgi([p for p in psi.T])
        for dim in range(psi.shape[1]):
            values_under_kde[psi[:, dim] < self.minimum[dim]] = 0.0
            values_under_kde[psi[:, dim] > self.maximum[dim]] = 0.0
        values_under_kde[values_under_kde == 0.0] = 1e-16
        return values_under_kde / 1000.0

    def log_prob(self, psi: Tensor, theta: Tensor):
        if self.rgi is None:
            self.build_kde()

        return torch.log(self.eval_kde(psi))


class KDE:
    """
    KDE where the returned log-prob is based on the nearest neighbour.
    """

    def __init__(self, prior_psi, prior_theta):
        self.kde = None
        self.prior_psi = prior_psi
        self.prior_theta = prior_theta

    def update_state(self, **kwargs):
        self.build_kde(**kwargs)

    def build_kde(self, num_samples: int = 1_000_000, num_bins: int = 100, **kwargs):
        theta_samples = self.prior_theta.sample((num_samples,))
        psi_samples = self.prior_psi.sample((num_samples,), theta=theta_samples)

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
        kde_values[kde_values == 0.0] = 1e-16
        return kde_values / 1000.0

    def log_prob(self, psi: Tensor, theta: Tensor):
        if self.kde is None:
            self.build_kde()

        return torch.log(self.eval_kde(psi))
