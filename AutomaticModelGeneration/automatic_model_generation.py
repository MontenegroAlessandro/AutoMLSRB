# Here we implement different versions of Algorithm Selection Modules
import numpy as np

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
                 trials_per_step=1, n_jobs=1, parallel_arms=1):
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

        if parallel_arms > len(arm_dictionary):
            parallel_arms = len(arm_dictionary)

        # set attributes from input
        self.budget = budget
        self.X = train_data_input
        self.Y = train_data_output
        self.trials_per_step = trials_per_step
        self.n_jobs = n_jobs
        self.parallel_arms = parallel_arms

        # define the arms dictionary to have correct key values
        self.arms = {}
        cnt = 0
        for key in arm_dictionary:
            self.arms[cnt] = arm_dictionary[key]
            cnt += 1

        # set additional attributes
        self.best_model = None
        self.best_model_eval = None
        self.last_pull = None
        self.step_id = 0

    def learn(self):
        # cycle over the arms and pull them
        for _ in range(self.budget):
            # select the next arm to pull
            to_pull = self.step_id % len(self.arms)

            # pull the arm
            print("[Log] To pull: ", to_pull)
            self.arms[to_pull].tuner.tune(self.trials_per_step)

            # update the best
            configurations = list(self.arms[to_pull].tuner.hpoptimizer.runhistory.config_ids.keys())
            costs = [self.arms[to_pull].tuner.hpoptimizer.runhistory.get_cost(config) for config in configurations]
            incumbent = configurations[np.argmin(costs)]
            incumbent_cost = np.min(costs)
            if self.best_model is None or self.best_model_eval >= incumbent_cost:
                print("[Log] New best: ", to_pull, " score: ", 1-incumbent_cost)
                self.best_model = self.arms[to_pull].model(**incumbent)
                self.best_model_eval = incumbent_cost

            # todo: save results in a json + save the current best model (???)

            # update the step_id and the last pull
            self.step_id += 1
            self.last_pull = to_pull

        # fit the model on the data
        model = self.best_model.fit(self.X, self.Y)
        return model


# Class implementing Algorithm Selection via Stochastic Rising Bandits
class StochasticRB:
    def __init__(self):
        pass

    def learn(self):
        pass

    def save_results(self):
        pass

    def reset(self):
        pass