# Here are present all the classes implementing Hyper Parameter Optimization procedures
import json
# Libraries
import os
import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
from ConfigSpace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.tae import StatusType
from smac.stats.stats import Stats


# Base Class
class BaseHPO(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def tune(self):
        pass

    @abstractmethod
    def save_results(self):
        pass


# Class implementing Bayesian Optimization via SMAC
class TunerSMAC(BaseHPO):
    def __init__(self, hp_dict=None, objective_foo=None, trials=1, log_path=None, n_jobs=1, seed=1):
        """
        :param hp_dict: a dictionary of hyperparameters to tune
        :param objective_foo: the objective function to optimize
        :param trials: how many trials to perform in a hpo step
        """
        # super class initializer
        super().__init__()

        # class attributes
        assert hp_dict is not None, "[Error] No hyperparameters passed."
        self.hp_dict = deepcopy(hp_dict)

        assert objective_foo is not None, "[Error] The objective function is None."
        self.objective_foo = objective_foo

        assert trials >= 1, "[Error] The number of trials is negative."
        self.trials = trials

        assert log_path is not None, "[Error] The log path is not specified."
        self.log_path = log_path

        assert n_jobs > 0, "[Error] Negative parallelism"
        self.n_jobs = n_jobs

        # flag spotting if it is the first run or not
        self.first_tune = True

        # other useful parameters
        self.n_runs = 1
        self.seed = seed
        self.run_history = None
        self.stats = None
        self.incumbent = None

        # build the configuration space
        self.config_space = ConfigurationSpace()
        self.config_space.seed(self.seed)
        for key in self.hp_dict:
            self.config_space.add_hyperparameter(self.hp_dict[key])

        # build the scenario
        self.scenario_dict = {
            "run_obj": "quality",
            "runcount_limit": self.trials,
            "cs": self.config_space,
            "deterministic": True,
            "output_dir": log_path
        }
        self.scenario = Scenario(self.scenario_dict)

        # create the SMAC object
        self.smac = SMAC4HPO(
            scenario=self.scenario,
            tae_runner=self.objective_foo,
            n_jobs=self.n_jobs,
            run_id=self.n_runs
        )

    def tune(self, trials=None):
        """
        :param trials: how many additional trials to perform, a trial is an evaluation of the objective function
        :return: the hyperparameter configuration with the minimum cost
        """
        # just run optimize if it is the first run
        if self.first_tune:
            self.first_tune = False

            # if the user passes a new number of trials, re-instantiate the smac object
            if trials is not None:
                self.trials = trials
                self.scenario_dict['runcount_limit'] = self.trials
                self.scenario = Scenario(self.scenario_dict)

                # create the SMAC object
                self.smac = SMAC4HPO(
                    scenario=self.scenario,
                    tae_runner=self.objective_foo,
                    n_jobs=self.n_jobs,
                    run_id=self.n_runs
                )
        # otherwise we need to restore the old context
        else:
            # Instantiate new SMAC run
            if trials is None:
                # when no trials are added, we just perform the double of the old number of trials
                trials = self.trials

            # Create scenario
            self.trials += trials
            self.scenario_dict['runcount_limit'] = self.trials
            self.scenario = Scenario(self.scenario_dict)

            # build the starting poit for the SMAC object
            self.build_last_history()

            # Now we can initialize SMAC with the recovered objects and restore the
            # state where we left off. By providing stats and a restore_incumbent, SMAC
            # automatically detects the intention of restoring a state.
            self.smac = SMAC4HPO(
                scenario=self.scenario,
                tae_runner=self.objective_foo,
                runhistory=self.run_history,
                stats=self.stats,
                restore_incumbent=self.incumbent,
                run_id=self.n_runs,
                n_jobs=self.n_jobs
            )

        # optimize
        res = self.smac.optimize()

        # save results
        self.save_results()

        # increment the number of runs
        self.n_runs += 1

        return res

    def save_results(self):
        file_name = os.path.join(self.log_path, "results.json")
        self.smac.runhistory.save_json(file_name)

    def build_last_history(self):
        # Populate run_history with custom data (e.g. from DataFrame)
        self.run_history = RunHistory()
        configurations = list(self.smac.runhistory.config_ids.keys())
        costs = [self.smac.runhistory.get_cost(config) for config in configurations]
        times = [0.] * len(costs)  # add your runtimes if applicable
        status = [StatusType.SUCCESS] * len(costs)
        self.incumbent = configurations[np.argmin(costs)]
        for i in range(len(configurations)):
            self.run_history.add(
                config=configurations[i],  # must be of type Configuration
                cost=costs[i],
                time=times[i],
                status=status[i],
                seed=self.seed
            )

        # Populate stats
        self.stats = Stats(self.scenario)
        keys = ["submitted_ta_runs", "finished_ta_runs"]
        n_points = len(self.run_history.data)
        for key in keys:
            setattr(self.stats, key, n_points)


# Class implementing HPO via a Genetic Algorithm
class TunerGenetic(BaseHPO):
    def __init__(self):
        super().__init__()

    def tune(self):
        pass

    # todo
    def save_results(self):
        pass


# Class implementing HPO via a BOHB (Bayesian Optimization Hyper Band)
class TunerBOHB(BaseHPO):
    def __init__(self):
        super().__init__()

    def tune(self):
        pass

    # todo
    def save_results(self):
        pass
