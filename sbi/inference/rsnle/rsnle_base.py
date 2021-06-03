# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from abc import ABC
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Union
from warnings import warn

import torch
from torch import Tensor, optim, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from sbi import utils as utils
from sbi.inference import NeuralInference
from sbi.inference.posteriors.likelihood_latent_based_posterior import (
    LikelihoodLatentBasedPosterior,
)
from sbi.types import TorchModule
from sbi.utils import check_estimator_arg, validate_theta_and_x, x_shape_from_simulation
from sbi.utils.sbiutils import mask_sims_from_prior
from sbi.utils.sbiutils import get_simulations_since_round
from sbi.utils import (
    handle_invalid_x,
    warn_if_zscoring_changes_data,
    warn_on_invalid_x,
    warn_on_invalid_x_for_snpec_leakage,
)


class LikelihoodLatentEstimator(NeuralInference, ABC):
    def __init__(
        self,
        prior,
        prior_z,
        density_estimator: Union[str, Callable] = "nsf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        **unused_args
    ):
        r"""Base class for Sequential Neural Likelihood Estimation methods.

        Args:
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.
            unused_args: Absorbs additional arguments. No entries will be used. If it
                is not empty, we warn. In future versions, when the new interface of
                0.14.0 is more mature, we will remove this argument.

        See docstring of `NeuralInference` class for all other arguments.
        """

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
            **unused_args
        )

        self._z_roundwise = []
        self._prior_z = prior_z

        # As detailed in the docstring, `density_estimator` is either a string or
        # a callable. The function creating the neural network is attached to
        # `_build_neural_net`. It will be called in the first round and receive
        # thetas and xs as inputs, so that they can be used for shape inference and
        # potentially for z-scoring.
        check_estimator_arg(density_estimator)
        if isinstance(density_estimator, str):
            self._build_neural_net = utils.likelihood_latent_nn(model=density_estimator)
        else:
            self._build_neural_net = density_estimator

        # SNLE-specific summary_writer fields.
        self._summary.update({"mcmc_times": []})  # type: ignore

    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        z: Tensor,
        from_round: int = 0,
    ) -> "LikelihoodLatentEstimator":
        r"""
        Store parameters and simulation outputs to use them for later training.

        Data are stored as entries in lists for each type of variable (parameter/data).

        Stores $\theta$, $x$, prior_masks (indicating if simulations are coming from the
        prior or not) and an index indicating which round the batch of simulations came
        from.

        Args:
            theta: Parameter sets.
            x: Simulation outputs.
            z: Latent variables.
            from_round: Which round the data stemmed from. Round 0 means from the prior.
                With default settings, this is not used at all for `SNLE`. Only when
                the user later on requests `.train(discard_prior_samples=True)`, we
                use these indices to find which training data stemmed from the prior.

        Returns:
            NeuralInference object (returned so that this function is chainable).
        """

        theta, x = validate_theta_and_x(theta, x, training_device=self._device)

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)
        self._z_roundwise.append(z)
        self._prior_masks.append(mask_sims_from_prior(int(from_round), theta.size(0)))
        self._data_round_index.append(int(from_round))

        return self

    def get_simulations(
        self,
        starting_round: int = 0,
        exclude_invalid_x: bool = True,
        warn_on_invalid: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Returns all $\theta$, $x$, and prior_masks from rounds >= `starting_round`.

        If requested, do not return invalid data.

        Args:
            starting_round: The earliest round to return samples from (we start counting
                from zero).
            exclude_invalid_x: Whether to exclude simulation outputs `x=NaN` or `x=±∞`
                during training.
            warn_on_invalid: Whether to give out a warning if invalid simulations were
                found.

        Returns: Parameters, simulation outputs, prior masks.
        """

        theta = get_simulations_since_round(
            self._theta_roundwise, self._data_round_index, starting_round
        )
        x = get_simulations_since_round(
            self._x_roundwise, self._data_round_index, starting_round
        )
        z = get_simulations_since_round(
            self._z_roundwise, self._data_round_index, starting_round
        )
        prior_masks = get_simulations_since_round(
            self._prior_masks, self._data_round_index, starting_round
        )

        # Check for NaNs in simulations.
        is_valid_x, num_nans, num_infs = handle_invalid_x(x, exclude_invalid_x)
        # Check for problematic z-scoring
        warn_if_zscoring_changes_data(x)
        if warn_on_invalid:
            warn_on_invalid_x(num_nans, num_infs, exclude_invalid_x)
            warn_on_invalid_x_for_snpec_leakage(
                num_nans, num_infs, exclude_invalid_x, type(self).__name__, self._round
            )

        return theta[is_valid_x], x[is_valid_x], z[is_valid_x], prior_masks[is_valid_x]

    def train(
        self,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        learning_rate_z_net: float = 1e-3,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        exclude_invalid_x: bool = True,
        resume_training: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch_each_round: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
    ) -> nn.Module:
        r"""
        Train the density estimator to learn the distribution $p(x|\theta)$.

        Args:
            exclude_invalid_x: Whether to exclude simulation outputs `x=NaN` or `x=±∞`
                during training. Expect errors, silent or explicit, when `False`.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch_each_round: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Density estimator that has learned the distribution $p(x|\theta)$.
        """

        max_num_epochs = 2 ** 31 - 1 if max_num_epochs is None else max_num_epochs

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)
        # Load data from most recent round.
        self._round = max(self._data_round_index)
        theta, x, z, _ = self.get_simulations(
            start_idx, exclude_invalid_x, warn_on_invalid=True
        )

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(theta, x, z)

        train_loader, val_loader = self.get_dataloaders(
            dataset,
            training_batch_size,
            validation_fraction,
            resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )

        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        if self._neural_net is None or retrain_from_scratch_each_round:
            self._neural_net = self._build_neural_net(
                theta[self.train_indices], x[self.train_indices], z[self.train_indices]
            )
            self._x_shape = x_shape_from_simulation(x)
            self._theta_dim = theta.shape[1]
            assert (
                len(self._x_shape) < 3
            ), "SNLE cannot handle multi-dimensional simulator output."

        self._neural_net.to(self._device)
        if not resume_training:
            self.optimizer = optim.Adam(
                list(self._neural_net.parameters()),
                lr=learning_rate,
            )
            self.optimizer = optim.Adam(
                [
                    {
                        "params": self._neural_net.flow_given_theta_psi.parameters(),
                        "lr": learning_rate,
                    },
                    {
                        "params": self._neural_net.net_z.parameters(),
                        "lr": learning_rate_z_net,
                    },
                ],
            )
            self.epoch, self._val_log_prob = 0, float("-Inf")

        while self.epoch <= max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):

            # Train for a single epoch.
            self._neural_net.train()
            for batch in train_loader:
                self.optimizer.zero_grad()
                theta_batch, x_batch, z_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                    batch[2].to(self._device),
                )
                # Evaluate on x with theta as context.
                theta_z_batch = torch.cat([theta_batch, z_batch], dim=1)
                log_prob = self._neural_net.log_prob(x_batch, context=theta_z_batch)
                loss = -torch.mean(log_prob)
                loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.parameters(),
                        max_norm=clip_max_norm,
                    )
                self.optimizer.step()

            self.epoch += 1
            # if self.epoch % 5 == 0:
            #     print(
            #         "std",
            #         torch.exp(
            #             self._neural_net.net_z.net.bell_features._stds_of_weights
            #         ),
            #     )

            # Calculate validation performance.
            self._neural_net.eval()
            log_prob_sum = 0
            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch, z_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                        batch[2].to(self._device),
                    )
                    # Evaluate on x with theta as context.
                    theta_z_batch = torch.cat([theta_batch, z_batch], dim=1)
                    log_prob = self._neural_net.log_prob(x_batch, context=theta_z_batch)
                    log_prob_sum += log_prob.sum().item()
            # Take mean over all validation samples.
            self._val_log_prob = log_prob_sum / (
                len(val_loader) * val_loader.batch_size
            )
            # Log validation log prob for every epoch.
            self._summary["validation_log_probs"].append(self._val_log_prob)

            self._maybe_show_progress(self._show_progress_bars, self.epoch)

        self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)

        # Update summary.
        self._summary["epochs"].append(self.epoch)
        self._summary["best_validation_log_probs"].append(self._best_val_log_prob)

        # Update TensorBoard and summary dict.
        self._summarize(
            round_=self._round,
            x_o=None,
            theta_bank=theta,
            x_bank=x,
        )

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        return deepcopy(self._neural_net)

    def build_posterior(
        self,
        density_estimator: Optional[TorchModule] = None,
        sample_with: str = "mcmc",
        mcmc_method: str = "slice_np",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        sample_z_given_psi_parameters: Optional[Dict[str, Any]] = None,
        psi_prior_eval_method: str = "kde_interpolate",
        psi_prior_eval_parameters: Dict = {},
    ) -> LikelihoodLatentBasedPosterior:
        r"""
        Build posterior from the neural density estimator.

        SNLE trains a neural network to approximate the likelihood $p(x|\theta)$. The
        `LikelihoodBasedPosterior` class wraps the trained network such that one can
        directly evaluate the unnormalized posterior log probability
        $p(\theta|x) \propto p(x|\theta) \cdot p(\theta)$ and draw samples from the
        posterior with MCMC.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`mcmc` | `rejection`].
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
                `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
                implementation of slice sampling; select `hmc`, `nuts` or `slice` for
                Pyro-based sampling.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior` will
                draw init locations from prior, whereas `sir` will use
                Sequential-Importance-Resampling using `init_strategy_num_candidates`
                to find init locations.
            rejection_sampling_parameters: Dictionary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `proposal` as the proposal distribtution (default is the prior).
                `max_sampling_batch_size` as the batchsize of samples being drawn from
                the proposal at every iteration. `num_samples_to_find_max` as the
                number of samples that are used to find the maximum of the
                `potential_fn / proposal` ratio. `num_iter_to_find_max` as the number
                of gradient ascent iterations to find the maximum of that ratio. `m` as
                multiplier to that ratio.

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods
            (the returned log-probability is unnormalized).
        """

        if density_estimator is None:
            density_estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        else:
            # Otherwise, infer it from the device of the net parameters.
            device = next(density_estimator.parameters()).device

        self._posterior = LikelihoodLatentBasedPosterior(
            method_family="snle",
            neural_net=density_estimator,
            prior=self._prior,
            prior_z=self._prior_z,
            x_shape=self._x_shape,
            theta_dim=self._theta_dim,
            sample_with=sample_with,
            mcmc_method=mcmc_method,
            mcmc_parameters=mcmc_parameters,
            rejection_sampling_parameters=rejection_sampling_parameters,
            sample_z_given_psi_parameters=sample_z_given_psi_parameters,
            psi_prior_eval_method=psi_prior_eval_method,
            psi_prior_eval_parameters=psi_prior_eval_parameters,
            device=device,
        )

        self._posterior._num_trained_rounds = self._round + 1

        # Store models at end of each round.
        self._model_bank.append(deepcopy(self._posterior))
        self._model_bank[-1].net.eval()

        return deepcopy(self._posterior)
