# Here are present all the classes implementing Hyper Parameter Optimization procedures

# Libraries
import os
import numpy as np
import errno, io, json
from abc import ABC, abstractmethod
from copy import deepcopy
from joblib import Parallel, delayed
from ConfigSpace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.facade.smac_mf_facade import SMAC4MF
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
    def __init__(self, hp_dict=None, objective_foo=None, trials=1, log_path=None, n_jobs=1, seed=1, conditions=None):
        """
        :param hp_dict: a dictionary of hyperparameters to tune
        :param objective_foo: the objective function to optimize
        :param trials: how many trials to perform in a hpo step
        """
        # super class initializer
        super().__init__()
        self.name = "smac"

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

        self.conditions = conditions

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

        if self.conditions is not None:
            self.config_space.add_conditions(self.conditions)

        # build the scenario
        self.scenario_dict = {
            "run_obj": "quality",
            "runcount_limit": self.trials,
            "cs": self.config_space,
            "deterministic": True,
            "output_dir": self.log_path
        }
        self.scenario = Scenario(self.scenario_dict)

        # create the SMAC object
        self.hpoptimizer = SMAC4HPO(
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
                self.hpoptimizer = SMAC4HPO(
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
            self.hpoptimizer = SMAC4HPO(
                scenario=self.scenario,
                tae_runner=self.objective_foo,
                runhistory=self.run_history,
                stats=self.stats,
                restore_incumbent=self.incumbent,
                run_id=self.n_runs,
                n_jobs=self.n_jobs
            )

        # optimize
        try:
            res = self.hpoptimizer.optimize()
        except ValueError:
            res = self.incumbent

        # save results
        self.save_results()

        # increment the number of runs
        self.n_runs += 1

        return res

    def save_results(self):
        file_name = os.path.join(self.log_path, "results_" + self.name + ".json")
        self.hpoptimizer.runhistory.save_json(file_name)

    def build_last_history(self):
        # Populate run_history with custom data (e.g. from DataFrame)
        self.run_history = RunHistory()
        configurations = list(self.hpoptimizer.runhistory.config_ids.keys())
        costs = [self.hpoptimizer.runhistory.get_cost(config) for config in configurations]
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
class TunerGenetic(BaseHPO, ABC):
    def __init__(self, n_agents=10, n_generations=100, prob_point_mutation=0.5, tuning_mode="best_performant_elitism",
                 pool_size=None, objective=None, hp_dict=None, seed=0, n_jobs=1, which_res="reward", log_path=None,
                 conditions=None):
        """
        :param n_agents: number of agents in each generation
        :param n_generations: number of generations
        :param prob_point_mutation: probability to mutate
        :param tuning_mode: "no_elitism", "best_performant_elitism", "pool_elitism"
        :param pool_size: the pool considered when "pool_elitism"
        """
        super().__init__()

        assert n_agents > 0
        self.n_agents = n_agents

        assert n_generations > 0
        self.n_generations = n_generations

        assert 0 <= prob_point_mutation <= 1
        self.prob_point_mutation = prob_point_mutation

        assert tuning_mode in ["no_elitism", "best_performant_elitism", "pool_elitism"]
        self.tuning_mode = tuning_mode

        if self.tuning_mode == "pool_elitism":
            self.pool_size = pool_size
            if (self.n_agents % self.pool_size) != 0:
                exc_msg = '\'n_agents\' must be an exact multiple of \'pool_size\'!'
                print(exc_msg)
                raise ValueError(exc_msg)
        else:
            self.pool_size = None

        assert objective is not None
        self.objective = objective

        self.seed = seed
        self.local_prng = np.random.default_rng(self.seed)
        self.n_jobs = n_jobs

        assert isinstance(hp_dict, dict)
        self.hp_dict = hp_dict
        self.cs = ConfigurationSpace(seed=seed)
        for hp in self.hp_dict:
            self.cs.add_hyperparameter(self.hp_dict[hp])
        if conditions != None:
            self.cs.add_conditions(conditions)

        assert which_res in ["reward", "cost"]
        self.which_res = which_res

        assert log_path is not None
        self.log_path = log_path
        self.name = "GeneticTuner"

        self.rolling_seed = 0
        self.best_agent = None
        self.best_agent_eval = None
        self.current_generation = None
        self.seen_config = []
        self.history = {}
        self.last_gen = None


    def tune(self, trials=None):
        if self.last_gen is None:
            # first generation
            agents = []
            agents_population = []

            # sample the configurations of hp
            for i in range(self.n_agents):
                tmp_ag = self._mutate(None)
                agents.append(deepcopy(tmp_ag))

            # evaluate those hp config
            parallel_agents_res = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=0, prefer="processes")
            parallel_agents_res = parallel_agents_res(
                delayed(self.objective)(agents[agent_index]["config"]) for agent_index in range(self.n_agents)
            )
            for i, res in enumerate(parallel_agents_res):
                if self.which_res == "reward":
                    agents[i]["block_eval"] = res
                else:
                    agents[i]["block_eval"] = 1 - res
                agents_population.append(deepcopy(agents[i]))
        else:
            self.n_generations = trials
            agents_population = deepcopy(self.last_gen)

        self._update_best(pop=deepcopy(agents_population))
        self.history["initial"] = deepcopy(agents_population)
        self.save_results()

        if self.tuning_mode == "no_elitism":
            final_pop = self._elitism(agents_population=deepcopy(agents_population), preserve_best=False)
        elif self.tuning_mode == "best_performant_elitism":
            final_pop = self._elitism(agents_population=deepcopy(agents_population), preserve_best=True)
        else:
            final_pop = self._pool_elitism(agents_population=deepcopy(agents_population))

        print("Best ag: " + str(self.best_agent))
        print("Best res: " + str(self.best_agent_eval))
        self.last_gen = deepcopy(final_pop)
        return self.best_agent

    def _elitism(self, agents_population=None, preserve_best=True):
        for gen_index in range(1, self.n_generations):
            print("[Log] Generation: " + str(gen_index + 1))
            new_agents_population = []
            tmp_new_agents_population = []

            while len(tmp_new_agents_population) < self.n_agents:
                # if it is the start of the generation i select the best agent of the previous generation to be passed
                # on:
                if len(tmp_new_agents_population) == 0:
                    selected_agent = self._evaluate_a_generation(gen=deepcopy(agents_population))[0]
                    log_msg = 'Previous generation best agent evaluation: ' + str(selected_agent["block_eval"])
                    print(log_msg)
                else:
                    # if the new generation is not empty I already copied the best agent, so I pick a new one with
                    # self.select(). I deepcopy the agents_population since I do not want to destroy it with
                    # self.select()
                    selected_agent = self._select(pop=deepcopy(agents_population))
                tmp_new_agents_population.append(deepcopy(selected_agent))

            agents = []
            # mutation process
            for i in range(self.n_agents):
                tmp_ag = self._mutate(deepcopy(tmp_new_agents_population[i]))
                agents.append(deepcopy(tmp_ag))
            # evaluate those hp config
            parallel_agents_res = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=0, prefer="processes")
            parallel_agents_res = parallel_agents_res(
                delayed(self.objective)(agents[agent_index]["config"]) for agent_index in range(self.n_agents)
            )

            if preserve_best:
                new_agents_population.append(deepcopy(self.best_agent))
                print("best appending")

            worst = 1
            w_id = None
            for i, res in enumerate(parallel_agents_res):
                if self.which_res == "reward":
                    agents[i]["block_eval"] = res
                else:
                    agents[i]["block_eval"] = 1 - res
                if preserve_best and agents[i]["block_eval"] <= worst:
                    w_id = i+1
                new_agents_population.append(deepcopy(agents[i]))
                print("appending " + str(i) + " with eval " + str(agents[i]["block_eval"]))

            if preserve_best:
                final_pop_current = []
                for i, elem in enumerate(new_agents_population):
                    if i != w_id:
                        final_pop_current.append(deepcopy(elem))
                new_agents_population = deepcopy(final_pop_current)

            agents_population = deepcopy(new_agents_population)
            self.history[gen_index] = deepcopy(agents_population)
            self._update_best(pop=deepcopy(agents_population))
            self.save_results()
            print("Generation End")

        return agents_population

    def _pool_elitism(self, agents_population=None):
        # get the indexes of the pool_size best agents
        reward_list = []
        for agent in agents_population:
            reward_list.append(agent["block_eval"])
        reward_list = np.array(reward_list)
        pool_of_best = np.argsort(reward_list)[len(reward_list) - self.pool_size:]
        pool_of_worst = np.argsort(reward_list)[:self.pool_size]

        for gen_index in range(self.n_generations - 1):
            print("Generation: " + str(gen_index + 1))

            new_agents_population = []
            tmp_new_agents_population = []

            # preserve the best agents
            for best_idx in pool_of_best:
                tmp_new_agents_population.append(deepcopy(agents_population[best_idx]))

            # complete the population via tournament selection
            while len(tmp_new_agents_population) < self.n_agents:
                selected_agent = self._select(pop=deepcopy(agents_population))
                tmp_new_agents_population.append(deepcopy(selected_agent))

            agents = []
            # mutation process
            for i in range(self.n_agents):
                tmp_ag = self._mutate(deepcopy(tmp_new_agents_population[i]))
                agents.append(deepcopy(tmp_ag))
            # evaluate those hp config
            parallel_agents_res = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=0, prefer="processes")
            parallel_agents_res = parallel_agents_res(
                delayed(self.objective)(agents[agent_index]["config"]) for agent_index in range(self.n_agents)
            )

            # preserve the best old pool of best
            for best_idx in pool_of_best:
                new_agents_population.append(deepcopy(agents_population[best_idx]))

            for i, res in enumerate(parallel_agents_res):
                if self.which_res == "reward":
                    agents[i]["block_eval"] = res
                else:
                    agents[i]["block_eval"] = 1 - res
                new_agents_population.append(deepcopy(agents[i]))

            reward_list = []
            for elem in new_agents_population:
                reward_list.append(elem["block_eval"])
            pool_of_worst = np.argsort(reward_list)[:self.pool_size]

            agents_population = []
            for id, ag in enumerate(new_agents_population):
                if id not in pool_of_worst:
                    agents_population.append(deepcopy(ag))

            self.history[gen_index] = deepcopy(agents_population)
            self._update_best(pop=deepcopy(agents_population))
            self.save_results()

        return agents_population

    def _select(self, pop):
        size_agents = 3
        tmp_gen = list(self.local_prng.choice(pop, size=size_agents))
        agent_to_pass_on = self._evaluate_a_generation(gen=tmp_gen)
        selected_ag = agent_to_pass_on[0]
        return selected_ag

    def _evaluate_a_generation(self, gen):
        if gen is None:
            self.is_tune_successful = False
            exc_msg = '\'gen\' cannot be \'None\'!'
            print(exc_msg)
            raise ValueError(exc_msg)

        agents_list_of_evaluations = []
        for tmp_agent in gen:
            agents_list_of_evaluations.append(tmp_agent["block_eval"])

        sign_for_sorting = -1

        best_agent_idx = np.argsort(sign_for_sorting * np.array(agents_list_of_evaluations))[0]

        best_agent = gen[best_agent_idx]
        best_agent_eval = best_agent["block_eval"]

        return best_agent, best_agent_eval

    def _update_best(self, pop=None):
        agents_list_of_evaluations = []
        for tmp_agent in pop:
            if tmp_agent["block_eval"] is not None:
                agents_list_of_evaluations.append(tmp_agent["block_eval"])
            else:
                agents_list_of_evaluations.append(0)

        sign_for_sorting = -1
        best_agent_idx = np.argsort(sign_for_sorting * np.array(agents_list_of_evaluations))[0]

        best_agent = pop[best_agent_idx]
        best_agent_eval = best_agent["block_eval"]

        if self.best_agent is None or self.best_agent_eval < best_agent_eval:
            print("[Debug] New Best, score: " + str(best_agent_eval))
            self.best_agent = deepcopy(best_agent)
            self.best_agent_eval = deepcopy(best_agent_eval)
        else:
            print("[Debug] Not new best")

    def _mutate(self, agent=None):
        if agent is None:
            agent = dict(
                config=None,
                block_eval=0
            )
            first_mutation = True
        else:
            first_mutation = False

        if self.local_prng.uniform() < self.prob_point_mutation or first_mutation:
            while True:
                tmp_config = self.cs.sample_configuration(1)

                if first_mutation:
                    old_config = {}
                else:
                    old_config = deepcopy(agent["config"])

                for hp in tmp_config:
                    if first_mutation:
                        old_config[hp] = deepcopy(tmp_config[hp])
                    elif self.local_prng.uniform() < self.prob_point_mutation:
                        old_config[hp] = deepcopy(tmp_config[hp])

                new_config = deepcopy(old_config)

                if tmp_config not in self.seen_config:
                    self.seen_config.append(deepcopy(tmp_config))

                    new_hps = {}
                    for key in new_config:
                        new_hps[key] = deepcopy(new_config[key])

                    agent["config"] = deepcopy(new_hps)
                    agent["block_eval"] = 0

                    break

        self.rolling_seed += 1
        self.cs.seed = self.seed + self.rolling_seed

        return deepcopy(agent)

    def save_results(self):
        name = self.log_path + "/" + self.name + "_results.json"

        if not os.path.exists(os.path.dirname(name)):
            try:
                os.makedirs(os.path.dirname(name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.history, ensure_ascii=False, indent=4))


# Class implementing HPO via a BOHB (Bayesian Optimization Hyper Band)
class TunerBOHB(BaseHPO):
    def __init__(self, hp_dict=None, objective_foo=None, trials=1, log_path=None, n_jobs=1, seed=1,
                 max_budget=2, initial_budget=1, eta=1, conditions=None):
        # superclass initialization
        super().__init__()
        self.name = "bohb"

        # class attributes
        assert hp_dict is not None, "[Error] No hyperparameters passed."
        self.hp_dict = deepcopy(hp_dict)

        assert objective_foo is not None, "[Error] The objective function is None."
        self.objective_foo = objective_foo

        assert trials >= 1, "[Error] The number of trials is negative."
        self.trials = trials

        assert log_path is not None, "[Error] The log path is not specified."
        self.log_path = log_path

        assert n_jobs >= -1, "[Error] Negative parallelism"
        self.n_jobs = n_jobs

        assert max_budget > 0, "[Error] Illegal value for max_budget."
        self.max_budget = max_budget

        assert initial_budget < max_budget, "[Error] Illegal value for initial budget."
        self.initial_budget = initial_budget

        assert eta > 0, "[Error] Illegal value for eta."
        self.eta = eta

        self.conditions = conditions

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

        if self.conditions is not None:
            self.config_space.add_conditions(self.conditions)

        # build the scenario
        self.scenario_dict = {
            "run_obj": "quality",
            "runcount_limit": self.trials,
            "cs": self.config_space,
            "deterministic": True,
            "output_dir": self.log_path
        }
        self.scenario = Scenario(self.scenario_dict)

        # create the SMAC object
        self.intensifier_kwargs = dict(
            max_budget=self.max_budget,
            initial_budget=self.initial_budget,
            eta=self.eta
        )
        self.hpoptimizer = SMAC4MF(
            scenario=self.scenario,
            tae_runner=self.objective_foo,
            n_jobs=self.n_jobs,
            run_id=self.n_runs,
            intensifier_kwargs=self.intensifier_kwargs
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
                self.hpoptimizer = SMAC4MF(
                    scenario=self.scenario,
                    tae_runner=self.objective_foo,
                    n_jobs=self.n_jobs,
                    run_id=self.n_runs,
                    intensifier_kwargs=self.intensifier_kwargs
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
            self.hpoptimizer = SMAC4MF(
                scenario=self.scenario,
                tae_runner=self.objective_foo,
                runhistory=self.run_history,
                stats=self.stats,
                restore_incumbent=self.incumbent,
                run_id=self.n_runs,
                n_jobs=self.n_jobs,
                intensifier_kwargs=self.intensifier_kwargs
            )

        # optimize
        try:
            res = self.hpoptimizer.optimize()
        except ValueError:
            res = self.incumbent

        # save results
        self.save_results()

        # increment the number of runs
        self.n_runs += 1

        return res

    def save_results(self):
        file_name = os.path.join(self.log_path, "results_" + self.name + ".json")
        self.hpoptimizer.runhistory.save_json(file_name)

    def build_last_history(self):
        # Populate run_history with custom data (e.g. from DataFrame)
        self.run_history = RunHistory()
        configurations = list(self.hpoptimizer.runhistory.config_ids.keys())
        costs = [self.hpoptimizer.runhistory.get_cost(config) for config in configurations]
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
