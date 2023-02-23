# Here we implement different versions of Algorithm Selection Modules
import math
import json
import os
import errno
import io
import numpy as np
from statistics import mean


# Class defining the Arm concept
class Arm:
    def __init__(self, model=None, tuner=None):
        assert model is not None, "[Error] No Model specified."
        self.model = model

        assert tuner is not None, "[Error] No Tuner specified."
        self.tuner = tuner


# Class implementing the base round-robin Algorithm Selection
class BaseAlgorithmSelection:
    def __init__(self, budget=None, train_data_input=None, train_data_output=None, arm_dictionary=None,
                 trials_per_step=1, n_jobs=1, parallel_arms=1, log_path=None):
        """
        :param budget: how many pulls (i.e., steps of HPO)
        :param train_data_input: the X
        :param train_data_output: the Y
        :param arm_dictionary: a dictionary of Arm objects
        :param trials_per_step: how many objective evaluations per optimization step
        :param n_jobs: how many parallel jobs per steps
        :param parallel_arms: how many parallel arms to select in each stage of the optimization procedure
        """
        # check over the parameters
        assert budget > 1, "[Error] Illegal Budget value."
        assert train_data_input is not None and train_data_output is not None, "[Error] No Training Data provided."
        assert arm_dictionary is not None, "[Error] No Arms provided."
        assert trials_per_step >= 1, "[Error] Illegal Number of trials per step."
        assert n_jobs >= 1 and parallel_arms >= 1, "[Error] Illegal Parallelization parameters."
        assert log_path is not None, "[Error] Invalid Log Path."

        if parallel_arms > len(arm_dictionary):
            parallel_arms = len(arm_dictionary)

        # set attributes from input
        self.budget = budget
        self.X = train_data_input
        self.Y = train_data_output
        self.trials_per_step = trials_per_step
        self.n_jobs = n_jobs
        self.parallel_arms = parallel_arms
        self.log_path=log_path

        # define the arms dictionary to have correct key values
        self.arms = {}
        cnt = 0
        for key in arm_dictionary:
            self.arms[cnt] = arm_dictionary[key]
            cnt += 1

        # set additional attributes
        self.name = "RoundRobin"
        self.n_arms = len(self.arms)
        self.best_model = None
        self.best_model_eval = None
        self.last_pull = None
        self.step_id = 0
        self.results = dict()
        self.pull_scores = np.zeros(self.budget)
        self.pulled_arms = np.zeros(self.budget)
        self.recommendation = None

    def learn(self):
        # cycle over the arms and pull them
        for _ in range(self.budget):
            # select the next arm to pull
            self.last_pull = self.step_id % len(self.arms)
            self.pulled_arms[self.step_id] = self.last_pull

            # pull the arm
            print("[Log] To pull: ", self.last_pull)
            self.arms[self.last_pull].tuner.tune(self.trials_per_step)

            # update the best
            self.update_best()

            # save results
            self.results = dict(
                pulled_arms=self.pulled_arms.tolist(),
                scores=self.pull_scores.tolist(),
                recommendation=self.recommendation
            )
            self.save_res()

            # update the step_id
            self.step_id += 1

        # fit the model on the data
        model = self.best_model.fit(self.X, self.Y)
        return model

    def update_best(self):
        """
        Function updating the best configuration found so far.
        :return: the mean of the observed rewards
        """
        configurations = list(self.arms[self.last_pull].tuner.hpoptimizer.runhistory.config_ids.keys())
        costs = [self.arms[self.last_pull].tuner.hpoptimizer.runhistory.get_cost(config) for config in configurations]
        incumbent = configurations[np.argmin(costs)]
        incumbent_cost = np.min(costs)
        if self.best_model is None or self.best_model_eval > incumbent_cost:
            print("[Log] New best: ", self.last_pull, " score: ", 1 - incumbent_cost)
            self.best_model = self.arms[self.last_pull].model(**incumbent)
            self.best_model_eval = incumbent_cost
            self.recommendation = self.last_pull
        costs = [self.arms[self.last_pull].tuner.hpoptimizer.runhistory.get_cost(config) for config in configurations
                 if self.arms[self.last_pull].tuner.hpoptimizer.runhistory.get_cost(config) != 0]
        if len(costs) - self.trials_per_step >= 0:
            mean_cost = mean(costs[len(costs) - self.trials_per_step:])
        else:
            mean_cost = mean(costs)
        print("[Log] Reward: ", 1 - mean_cost)
        self.pull_scores[self.step_id] = 1 - mean_cost
        return 1 - mean_cost

    def save_res(self):
        name = self.log_path + "/" + self.name + "_results.json"

        if not os.path.exists(os.path.dirname(name)):
            try:
                os.makedirs(os.path.dirname(name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.results, ensure_ascii=False, indent=4))


# Class implementing Algorithm Selection via Stochastic Rising Bandits
class AlgorithmSelectionSRB(BaseAlgorithmSelection):
    def __init__(self, exp_param=1, sigma=0.1, eps=0.25, budget=None, train_data_input=None, train_data_output=None,
                 arm_dictionary=None,
                 trials_per_step=1, n_jobs=1, parallel_arms=1, log_path=None):
        """
        :param exp_param: exploration parameter
        :param sigma: expected noise
        :param eps: epsilon parameter to select the window size
        :param budget: how many pulls (i.e., steps of HPO)
        :param train_data_input: the X
        :param train_data_output: the Y
        :param arm_dictionary: a dictionary of Arm objects
        :param trials_per_step: how many objective evaluations per optimization step
        :param n_jobs: how many parallel jobs per steps
        :param parallel_arms: how many parallel arms to select in each stage of the optimization procedure
        """
        # initialize the super class
        super().__init__(budget=budget, train_data_input=train_data_input, train_data_output=train_data_output,
                         arm_dictionary=arm_dictionary, trials_per_step=trials_per_step, n_jobs=n_jobs,
                         parallel_arms=parallel_arms, log_path=log_path)

        # check additional input
        assert exp_param >= 0, "[Error] Illegal Exploration Parameter."
        assert sigma >= 0, "[Error] Illegal negative value for the expected Noise."
        assert 0 < eps < 0.5, "[Error] Illegal value for Epsilon. Remember eps in (0, 0.5)."

        # set attributes from inputs
        self.exp_param = exp_param
        self.sigma = sigma
        self.eps = eps

        # set additional attributes
        self.name = "RUCB"
        self.n_arms = len(self.arms)
        self.warmup = True
        self.pulls = np.zeros(self.n_arms)
        self.window = np.zeros(self.n_arms)
        self.mu_check = np.zeros(self.n_arms)
        self.beta_check = np.zeros(self.n_arms)
        self.upper_bound = np.inf * np.ones(self.n_arms)
        self.a = np.zeros(self.n_arms)
        self.b = np.zeros(self.n_arms)
        self.c = np.zeros(self.n_arms)
        self.d = np.zeros(self.n_arms)
        self.scores = np.zeros((self.n_arms, self.budget))

    def learn(self):
        for _ in range(self.budget):
            print("[Log] Step: ", self.step_id)
            # select the arm to pull
            if self.warmup:
                self.last_pull = self.step_id % self.n_arms
                print("[Log] Pull: ", self.last_pull)
            else:
                self.last_pull = np.argmax(self.upper_bound)
                print("[Log] Pull: ", self.last_pull)
                print("[Log] UBs: ", self.upper_bound)
            self.pulled_arms[self.step_id] = self.last_pull

            # pull the arm
            self.arms[self.last_pull].tuner.tune(self.trials_per_step)

            # update the best
            reward = self.update_best()

            # save results
            self.results = dict(
                pulled_arms=self.pulled_arms.tolist(),
                scores=self.pull_scores.tolist(),
                recommendation=self.recommendation
            )
            self.save_res()

            # update the parameters
            self.step_id += 1
            arm = int(self.last_pull)
            self.pulls[arm] += 1

            n = int(self.pulls[arm])
            h = math.floor(self.eps * n)

            self.scores[arm][n - 1] = reward

            if h == self.window[arm]:
                self.a[arm] += reward - self.scores[arm][n - h - 1]
                self.b[arm] += self.scores[arm][n - h - 1] - self.scores[arm][n - 2 * h - 1]
                self.c[arm] += n * reward - (n - h) * self.scores[arm][n - h - 1]
                self.d[arm] += n * self.scores[arm][n - h - 1] - (n - h) * self.scores[arm][n - 2 * h - 1]
            else:
                self.a[arm] += reward
                self.b[arm] += self.scores[arm][n - 2 * h]
                self.c[arm] += n * reward
                self.d[arm] += (n - h) * self.scores[arm][n - 2 * h] + self.b[arm]

            self.window[arm] = h
            a = self.a[arm]
            b = self.b[arm]
            c = self.c[arm]
            d = self.d[arm]

            projection_point = self.budget - self.step_id + n

            self.mu_check[arm] = (1 / h) * (a + (projection_point * (a - b) / h) - ((c - d) / h)) if h > 0 else 0
            self.beta_check[arm] = self.sigma * (projection_point - n + h - 1) * math.sqrt(
                (self.exp_param) / (math.pow(h, 3))) if h > 0 else 0
            self.upper_bound[arm] = self.mu_check[arm] + self.beta_check[arm] if h > 0 else 0

            # check if the warmup phase is over
            if 0 not in self.window:
                self.warmup = False

        # fit the model on the data
        model = self.best_model.fit(self.X, self.Y)
        return model

    def save_results(self):
        pass


class AlgorithmSelectionAdaptiveSRB(BaseAlgorithmSelection):
    def __init__(self, exp_param=1, eps=0.25, budget=None, train_data_input=None, train_data_output=None,
                 arm_dictionary=None,
                 trials_per_step=1, n_jobs=1, parallel_arms=1, log_path=None):
        """
        AlgorithmSelectionSRB module with online learning of sigma.
        :param exp_param: exploration parameter
        :param eps: epsilon parameter to select the window size
        :param budget: how many pulls (i.e., steps of HPO)
        :param train_data_input: the X
        :param train_data_output: the Y
        :param arm_dictionary: a dictionary of Arm objects
        :param trials_per_step: how many objective evaluations per optimization step
        :param n_jobs: how many parallel jobs per steps
        :param parallel_arms: how many parallel arms to select in each stage of the optimization procedure
        """
        # initialize the super class
        super().__init__(budget=budget, train_data_input=train_data_input, train_data_output=train_data_output,
                         arm_dictionary=arm_dictionary, trials_per_step=trials_per_step, n_jobs=n_jobs,
                         parallel_arms=parallel_arms, log_path=log_path)

        # check additional input
        assert exp_param >= 0, "[Error] Illegal Exploration Parameter."
        assert 0 < eps < 0.5, "[Error] Illegal value for Epsilon. Remember eps in (0, 0.5)."

        # set attributes from inputs
        self.exp_param = exp_param
        self.eps = eps

        # set additional attributes
        self.name = "AdaptiveRUCB"
        self.n_arms = len(self.arms)
        self.warmup = True
        self.pulls = np.zeros(self.n_arms)
        self.window = np.zeros(self.n_arms)
        self.mu_check = np.zeros(self.n_arms)
        self.beta_check = np.zeros(self.n_arms)
        self.upper_bound = np.inf * np.ones(self.n_arms)
        self.a = np.zeros(self.n_arms)
        self.b = np.zeros(self.n_arms)
        self.c = np.zeros(self.n_arms)
        self.d = np.zeros(self.n_arms)
        self.scores = np.zeros((self.n_arms, self.budget))

        # The sigma parameter is initialized with the standard deviation of the Uniform-Distribution
        self.sigma = 0.5 * np.ones(self.n_arms)
        self.sigmas = np.zeros((self.budget, self.n_arms))

    def learn(self):
        for _ in range(self.budget):
            print("[Log] Step: ", self.step_id)
            # select the arm to pull
            if self.warmup:
                self.last_pull = self.step_id % self.n_arms
                print("[Log] Pull: ", self.last_pull)
            else:
                self.last_pull = np.argmax(self.upper_bound)
                print("[Log] Pull: ", self.last_pull)
                print("[Log] UBs: ", self.upper_bound)
            print("[Log] Sigma: ", self.sigma)
            self.pulled_arms[self.step_id] = self.last_pull

            # pull the arm
            self.arms[self.last_pull].tuner.tune(self.trials_per_step)

            # update the best
            reward = self.update_best()

            # update the parameters
            # self.step_id += 1
            arm = int(self.last_pull)
            self.pulls[arm] += 1

            n = int(self.pulls[arm])
            h = math.floor(self.eps * n)

            self.scores[arm][n - 1] = reward

            if h == self.window[arm]:
                self.a[arm] += reward - self.scores[arm][n - h - 1]
                self.b[arm] += self.scores[arm][n - h - 1] - self.scores[arm][n - 2 * h - 1]
                self.c[arm] += n * reward - (n - h) * self.scores[arm][n - h - 1]
                self.d[arm] += n * self.scores[arm][n - h - 1] - (n - h) * self.scores[arm][n - 2 * h - 1]
            else:
                self.a[arm] += reward
                self.b[arm] += self.scores[arm][n - 2 * h]
                self.c[arm] += n * reward
                self.d[arm] += (n - h) * self.scores[arm][n - 2 * h] + self.b[arm]

            self.window[arm] = h
            a = self.a[arm]
            b = self.b[arm]
            c = self.c[arm]
            d = self.d[arm]

            projection_point = self.budget - (self.step_id+1) + n

            self.mu_check[arm] = (1 / h) * (a + (projection_point * (a - b) / h) - ((c - d) / h)) if h > 0 else 0
            self.beta_check[arm] = self.sigma[arm] * (projection_point - n + h - 1) * math.sqrt(
                (self.exp_param) / (math.pow(h, 3))) if h > 0 else 0
            self.upper_bound[arm] = self.mu_check[arm] + self.beta_check[arm] if h > 0 else 0

            # update the sigma_i
            if h > 1:
                res = 0
                for j in range(n - h, n):
                    estimated_reward = self.scores[arm][j]
                    estimated_increment = (projection_point - j) * (estimated_reward - self.scores[arm][j - h]) / h
                    res += (estimated_reward + estimated_increment - self.mu_check[arm]) ** 2
                self.sigma[arm] = math.sqrt(res / (h - 1))
            self.sigmas[self.step_id] = self.sigma

            # check if the warmup phase is over
            if 0 not in self.window:
                self.warmup = False

            # save results
            self.results = dict(
                pulled_arms=self.pulled_arms.tolist(),
                scores=self.pull_scores.tolist(),
                recommendation=self.recommendation,
                sigmas=self.sigmas.tolist()
            )
            self.save_res()
            self.step_id += 1

        # fit the model on the data
        model = self.best_model.fit(self.X, self.Y)
        return model

    def save_results(self):
        pass


class EfficientCASHRB(BaseAlgorithmSelection):
    def __init__(self, budget=None, train_data_input=None, train_data_output=None, arm_dictionary=None,
                 trials_per_step=1, n_jobs=1, parallel_arms=1, log_path=None):
        # super class instantiation
        super().__init__(budget=budget, train_data_input=train_data_input, train_data_output=train_data_output,
                         arm_dictionary=arm_dictionary, trials_per_step=trials_per_step, n_jobs=n_jobs,
                         parallel_arms=parallel_arms, log_path=log_path)

        # set additional attributes
        self.name = "EfficientCash"
        self.upper_bounds = np.ones(self.n_arms)
        self.lower_bounds = np.zeros(self.n_arms)
        self.S_candidate = {}
        for i in range(self.n_arms):
            self.S_candidate[i] = i
        self.pulls = np.zeros(self.n_arms)
        self.scores = np.zeros((self.n_arms, self.budget))
        self.deleted = []

    def learn(self):
        # iterate over the budget
        while self.step_id < self.budget:
            # iterate over the remaining algorithms
            for arm in self.S_candidate:
                # increment the step id
                print("[Log] Step: ", self.step_id+1)
                print("[Log] Pulls: ", self.pulls)
                print("[Log] Scores: ", self.scores)
                print("[Log] UBs: ", self.upper_bounds)
                print("[Log] LBs: ", self.lower_bounds)
                print("[Log] Candidates: ", self.S_candidate)

                # pull the current arm
                self.last_pull = int(arm)
                self.pulled_arms[self.step_id] = self.last_pull
                self.arms[self.last_pull].tuner.tune(self.trials_per_step)

                # update the best and get the mean observation
                reward = self.update_best()

                # update the arm's parameters
                self.pulls[self.last_pull] += 1
                n = int(self.pulls[self.last_pull])
                self.scores[self.last_pull][n - 1] = reward

                # spot the rate, ub and lb
                weight = reward - self.lower_bounds[self.last_pull]
                self.upper_bounds[self.last_pull] = min(1, reward + weight*(self.budget - (self.step_id + 1)))
                self.lower_bounds[self.last_pull] = reward

                self.step_id += 1

            # elimination procedure
            to_del = []
            for i in self.S_candidate:
                if i not in to_del:
                    for j in self.S_candidate:
                        if j not in to_del:
                            if (j != i) and (self.lower_bounds[i] >= self.upper_bounds[j]):
                                to_del.append(j)

            for elem in to_del:
                del self.S_candidate[elem]
                self.deleted.append(elem)

            # save res
            self.results = dict(
                pulled_arms=self.pulled_arms.tolist(),
                scores=self.pull_scores.tolist(),
                recommendation=self.recommendation,
                deleted=self.deleted
            )
            self.save_res()
