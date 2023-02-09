import numpy as np
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from sklearn.ensemble import RandomForestClassifier
from HyperparametersOptimization.hyperparemeters_optimization import TunerSMAC


def train_random_forest(config):
    model = RandomForestClassifier(max_depth=config["depth"])
    model.fit(X_train, y_train)

    # Define the evaluation metric as return
    return 1 - model.score(X_val, y_val)


if __name__ == "__main__":
    X_train, y_train = np.random.randint(2, size=(20, 2)), np.random.randint(2, size=20)
    X_val, y_val = np.random.randint(2, size=(5, 2)), np.random.randint(2, size=5)

    hp_dict = {0: UniformIntegerHyperparameter("depth", 2, 100)}

    tuner = TunerSMAC(
        hp_dict=hp_dict,
        objective_foo=train_random_forest,
        trials=10,
        log_path="experiments/SMAC",
        n_jobs=1,
        seed=2023
    )

    res = tuner.tune()
    print(res)
    print(list(tuner.smac.runhistory.config_ids.keys()))
    print([tuner.smac.runhistory.get_cost(config) for config in list(tuner.smac.runhistory.config_ids.keys())])

    res = tuner.tune(30)
    print(res)
    print(list(tuner.smac.runhistory.config_ids.keys()))
    print([tuner.smac.runhistory.get_cost(config) for config in list(tuner.smac.runhistory.config_ids.keys())])