import copy
from scipy.stats import beta, bernoulli
import numpy as np


class MABBaseSampler:
    def __init__(self, K: int, priors: dict = {}):
        """Thompson Sampling implementation for a Bernoulli problem, using beta distributions
        as sampling distributions.

        Args:
            K (int): slot machines amount
            priors (dict, optional): Priors beliefs for 'alpha' and 'beta' parameters for
            every slot machine, it must respect the form:
            {slot_machine_number:{'alpha':float,'beta':float}}
            If some slot machines do not have beliefs, we recommend to set the parameters
            to .5. Defaults to {}, meanining no prior belief at all.
        """
        self.K = K
        if len(priors) != K:
            self.posteriors_params = {
                kth_mab: {"alpha": 0.5, "beta": 0.5} for kth_mab in range(K)
            }
        else:
            self.posteriors_params = priors
        self.posterior_params_tracker = [copy.deepcopy(self.posteriors_params)]

    def _sample_from_posterior(self, kth_mab_index: int) -> float:
        """Sample from the posterior of the selected slot machine

        Args:
            kth_mab_index (int): index of the selected slot machine to update
            posterior

        Returns:
            float: sample of the posterior distribution
        """
        beta_dist = beta(
            self.posteriors_params[kth_mab_index]["alpha"],
            self.posteriors_params[kth_mab_index]["beta"],
        )
        return beta_dist.rvs(size=1)[0]

    def calculate_step_regret(
        self, actual_probabilities: list, step: int, historical_rewards: list
    ) -> float:
        """Calculates regret for an specific step in the iteration:

        Let:
            * Rt be the regret for time step t
            * mu_star be the highest real slot machines probability
            * Xt be the reward at time step t
            * Ht be an array with each Xt

            Rt = t * mu_star - sum(Xt)

        Args:
            actual_probabilities (list): real probabilities of success for
            each slot machine
            step (int): current iteration step
            historical_rewards (list): list with historical rewards from t=0 to t=actual_step

        Returns:
            float: regret
        """
        return step * np.max(actual_probabilities) - np.sum(historical_rewards)

    def update_posterior(self, kth_mab_index: int, reward: int):
        """Update slot machine posterior distributions.

        Args:
            kth_mab_index (int): index of the selected slot machine to update
            posterior
            reward (int): takes the values 0 or 1, represents if the play
            was successful or not
        """
        self.posteriors_params[kth_mab_index]["alpha"] = (
            self.posteriors_params[kth_mab_index]["alpha"] + reward
        )
        self.posteriors_params[kth_mab_index]["beta"] = (
            self.posteriors_params[kth_mab_index]["beta"] + 1 - reward
        )
        self.posterior_params_tracker.append(copy.deepcopy(self.posteriors_params))

    def sample_from_real_distribution(
        self, actual_probabilities: list, kth_mab_index: int
    ) -> int:
        """Generate a sample from the real distribution to get a real reward

        Args:
            actual_probabilities (list): real probabilities of success for
            each slot machine
            kth_mab_index (int): index of the selected slot machine to update
            posterior

        Returns:
            int: reward, takes values 0 or 1
        """
        assert (
            len(actual_probabilities) == self.K
        ), "Must provide pobabilities for all the slot machines"
        return bernoulli(actual_probabilities[kth_mab_index]).rvs(size=1)[0]


class BernoulliTompsonSampling(MABBaseSampler):
    def __init__(self, K: int, priors: dict = {}):
        """Thompson Sampling implementation for a Bernoulli problem, using beta distributions
        as sampling distributions.

        Args:
            K (int): slot machines amount
            priors (dict, optional): Priors beliefs for 'alpha' and 'beta' parameters for
            every slot machine, it must respect the form:
            {slot_machine_number:{'alpha':float,'beta':float}}
            If some slot machines do not have beliefs, we recommend to set the parameters
            to .5. Defaults to {}, meanining no prior belief at all.
        """
        super().__init__(K, priors)

    def sample_mabs_posteriors(self) -> int:
        """Generate a sample for the posterior distribution for each slot machine
        and returns the slot machine index with the greatest realization

        Returns:
            int: index of the slot machine with the biggest sampled number
        """
        current_sampling = []
        for k in range(self.K):
            current_sampling.append(self._sample_from_posterior(k))
        return np.argmax(current_sampling)


class BernoulliEpsilonGreedy(MABBaseSampler):
    def __init__(self, K: int, epsilon: float, priors: dict = {}):
        super().__init__(K, priors)
        self.e = epsilon

    def explore(self):
        return np.random.choice(range(self.K))

    def exploit(self):
        expected_values = []
        for kth_mab_index in range(self.K):
            expected_values.append(
                self.posteriors_params[kth_mab_index]["alpha"]
                / (
                    self.posteriors_params[kth_mab_index]["alpha"]
                    + self.posteriors_params[kth_mab_index]["beta"]
                )
            )
        return np.argmax(expected_values)

    def select_mab(self):
        p = bernoulli(self.e)
        if p == 0:
            return self.exploit()
        else:
            return self.explore()


# class BernoulliTompsonSampling:
#     def __init__(self, K: int, priors: dict = {}):
#         """Thompson Sampling implementation for a Bernoulli problem, using beta distributions
#         as sampling distributions.

#         Args:
#             K (int): slot machines amount
#             priors (dict, optional): Priors beliefs for 'alpha' and 'beta' parameters for
#             every slot machine, it must respect the form:
#             {slot_machine_number:{'alpha':float,'beta':float}}
#             If some slot machines do not have beliefs, we recommend to set the parameters
#             to .5. Defaults to {}, meanining no prior belief at all.
#         """
#         self.K = K
#         if len(priors) != K:
#             self.posteriors_params = {
#                 kth_mab: {"alpha": 0.5, "beta": 0.5} for kth_mab in range(K)
#             }
#         else:
#             self.posteriors_params = priors
#         self.posterior_params_tracker = [copy.deepcopy(self.posteriors_params)]

#     def update_posterior(self, kth_mab_index: int, reward: int):
#         """Update slot machine posterior distributions.

#         Args:
#             kth_mab_index (int): index of the selected slot machine to update
#             posterior
#             reward (int): takes the values 0 or 1, represents if the play
#             was successful or not
#         """
#         self.posteriors_params[kth_mab_index]["alpha"] = (
#             self.posteriors_params[kth_mab_index]["alpha"] + reward
#         )
#         self.posteriors_params[kth_mab_index]["beta"] = (
#             self.posteriors_params[kth_mab_index]["beta"] + 1 - reward
#         )
#         self.posterior_params_tracker.append(copy.deepcopy(self.posteriors_params))

#     def _sample_from_posterior(self, kth_mab_index: int) -> float:
#         """Sample from the posterior of the selected slot machine

#         Args:
#             kth_mab_index (int): index of the selected slot machine to update
#             posterior

#         Returns:
#             float: sample of the posterior distribution
#         """
#         beta_dist = beta(
#             self.posteriors_params[kth_mab_index]["alpha"],
#             self.posteriors_params[kth_mab_index]["beta"],
#         )
#         return beta_dist.rvs(size=1)[0]

#     def sample_from_real_distribution(
#         self, actual_probabilities: list, kth_mab_index: int
#     ) -> int:
#         """Generate a sample from the real distribution to get a real reward

#         Args:
#             actual_probabilities (list): real probabilities of success for
#             each slot machine
#             kth_mab_index (int): index of the selected slot machine to update
#             posterior

#         Returns:
#             int: reward, takes values 0 or 1
#         """
#         assert (
#             len(actual_probabilities) == self.K
#         ), "Must provide pobabilities for all the slot machines"
#         return bernoulli(actual_probabilities[kth_mab_index]).rvs(size=1)[0]

#     def sample_mabs_posteriors(self) -> int:
#         """Generate a sample for the posterior distribution for each slot machine
#         and returns the slot machine index with the greatest realization

#         Returns:
#             int: index of the slot machine with the biggest sampled number
#         """
#         current_sampling = []
#         for k in range(self.K):
#             current_sampling.append(self._sample_from_posterior(k))
#         return np.argmax(current_sampling)

#     def calculate_step_regret(
#         self, actual_probabilities: list, step: int, historical_rewards: list
#     ) -> float:
#         """Calculates regret for an specific step in the iteration:

#         Let:
#             * Rt be the regret for time step t
#             * mu_star be the highest real slot machines probability
#             * Xt be the reward at time step t
#             * Ht be an array with each Xt

#             Rt = t * mu_star - sum(Xt)

#         Args:
#             actual_probabilities (list): real probabilities of success for
#             each slot machine
#             step (int): current iteration step
#             historical_rewards (list): list with historical rewards from t=0 to t=actual_step

#         Returns:
#             float: regret
#         """
#         return step * np.max(actual_probabilities) - np.sum(historical_rewards)
