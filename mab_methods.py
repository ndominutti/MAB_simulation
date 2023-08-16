import copy
from scipy.stats import beta, bernoulli
import numpy as np
from collections import Counter


class MABBaseSampler:
    def __init__(self, K: int, priors: dict = {}, random_seed: int = None):
        """Base sampler for MAB methods comparison.

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
                kth_mab: {"alpha": 1, "beta": 1} for kth_mab in range(K)
            }
        else:
            self.posteriors_params = priors
        self.posterior_params_tracker = [copy.deepcopy(self.posteriors_params)]
        if random_seed is not None:
            np.random.seed(random_seed)

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
        # If there's a tie, choose at random
        if np.all(np.array(current_sampling) == current_sampling[0]):
            return np.random.choice(range(self.K))
        else:
            return np.argmax(current_sampling)


class BernoulliEpsilonGreedy(MABBaseSampler):
    def __init__(self, K: int, epsilon: float, priors: dict = {}, decay: bool = False):
        super().__init__(K, priors)
        # Leave epsilon as an static value for decay
        self.epsilon = copy.deepcopy(epsilon)
        self.e = copy.deepcopy(epsilon)
        self.decay = decay

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
        # If there's a tie, choose at random
        if np.all(np.array(expected_values) == expected_values[0]):
            return np.random.choice(expected_values)
        else:
            return np.argmax(expected_values)

    def select_mab(self, step: int = None):
        p = bernoulli(self.e)
        if self.decay:
            assert (
                step is not None
            ), "If you want the epsilon to decay, you must pass the actual iteration step"
            self.e = self.epsilon / step
        if p == 0:
            return self.exploit()
        else:
            return self.explore()


class BernoulliUBC(MABBaseSampler):
    def __init__(self, K: int, priors: dict = {}):
        super().__init__(K, priors)

    def select_mab(self, step, historical_selections):
        selection_bounds = []
        historical_selection_counter = Counter(historical_selections)
        for kth_mab_index in range(self.K):
            expected_value = self.posteriors_params[kth_mab_index]["alpha"] / (
                self.posteriors_params[kth_mab_index]["alpha"]
                + self.posteriors_params[kth_mab_index]["beta"]
            )
            selection_bounds.append(
                expected_value
                + np.sqrt(
                    (2 * np.log(step)) / (historical_selection_counter[kth_mab_index])
                )
            )
        # If there's a tie, choose at random
        if np.all(np.array(selection_bounds) == selection_bounds[0]):
            return np.random.choice(range(self.K))
        else:
            return np.argmax(selection_bounds)
