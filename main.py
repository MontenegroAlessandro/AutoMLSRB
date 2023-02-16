# Libraries
import pickle
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from Datasets.ClsDatasets import WineDataset
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, Constant, UnParametrizedHyperparameter
from HyperparametersOptimization.hyperparemeters_optimization import TunerSMAC, TunerBOHB
from sklearn.model_selection import cross_val_score
from AutomaticModelSelection.automatic_model_selection import Arm, BaseAlgorithmSelection, AlgorithmSelectionSRB, \
    AlgorithmSelectionAdaptiveSRB
from copy import deepcopy

# Data
data = WineDataset()
X = data.input
Y = data.target
print(X.shape, Y.shape)

# Hyperparameter(s)
hp_dict_adaboost = dict(
    n_estimators=UniformIntegerHyperparameter(name="n_estimators", lower=50, upper=500, default_value=50),
    learning_rate=UniformFloatHyperparameter(name="learning_rate", lower=0.01, upper=3, default_value=0.1),
    algorithm=CategoricalHyperparameter(name="algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")
)

hp_dict_rf = dict(
    n_estimators=Constant("n_estimators", 100),
    criterion=CategoricalHyperparameter(name="criterion", choices=["gini", "entropy"], default_value="gini"),
    max_depth=UniformIntegerHyperparameter(name="max_depth", lower=1, upper=200, default_value=30),
    max_features=UniformFloatHyperparameter(name="max_features", lower=0., upper=1., default_value=0.5),
    min_samples_split=UniformIntegerHyperparameter(name="min_samples_split", lower=2, upper=20, default_value=2),
    min_samples_leaf=UniformIntegerHyperparameter(name="min_samples_leaf", lower=1, upper=20, default_value=1),
    min_weight_fraction_leaf=UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.),
    max_leaf_nodes=UnParametrizedHyperparameter("max_leaf_nodes", "None"),
    min_impurity_decrease=UnParametrizedHyperparameter('min_impurity_decrease', 0.0),
    bootstrap=CategoricalHyperparameter(name="bootstrap", choices=["True", "False"], default_value="True")
)


# Objective(s)
def objective_adaboost(config):
    model = AdaBoostClassifier(n_estimators=config["n_estimators"], learning_rate=config["learning_rate"],
                               algorithm=config["algorithm"])
    scores = cross_val_score(model, X, Y, cv=10)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()


def objective_rf(config):
    model = RandomForestClassifier(max_depth=config["max_depth"], criterion=config["criterion"],
                                   max_features=config["max_features"], min_samples_split=config["min_samples_split"],
                                   min_samples_leaf=config["min_samples_leaf"],
                                   min_weight_fraction_leaf=config["min_weight_fraction_leaf"],
                                   max_leaf_nodes=config["max_leaf_nodes"],
                                   min_impurity_decrease=config["min_impurity_decrease"], bootstrap=config["bootstrap"])
    scores = cross_val_score(model, X, Y, cv=10)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()


# Tuner(s)
base_dir = "experiments/Test1/"
tuner_args = dict(
    hp_dict=deepcopy(hp_dict_adaboost),
    objective_foo=objective_adaboost,
    trials=10,
    log_path=base_dir + "test_ada",
    n_jobs=1,
    seed=2023
    # max_budget=30,
    # initial_budget=10,
    # eta=3
)
# tuner_adaboost = TunerBOHB(**tuner_args)
tuner_adaboost = TunerSMAC(**tuner_args)

tuner_args["hp_dict"] = deepcopy(hp_dict_rf)
tuner_args["objective_foo"] = objective_rf
tuner_args["log_path"] = base_dir + "test_rf"
tuner_rf = TunerSMAC(**tuner_args)
# tuner_rf = TunerBOHB(**tuner_args)

# Arm(s)
arm_adaboost = Arm(model=AdaBoostClassifier, tuner=tuner_adaboost)
arm_rf = Arm(model=RandomForestClassifier, tuner=tuner_rf)

# Dictionary of Arm(s)
arms_dict = dict(
    adaboost=arm_adaboost,
    random_forest=arm_rf
)

# Automatic Block
auto_model_generation = BaseAlgorithmSelection(
    budget=10,
    train_data_input=X,
    train_data_output=Y,
    arm_dictionary=arms_dict,
    trials_per_step=10
)

auto_model_generation2 = AlgorithmSelectionSRB(
    exp_param=1,
    eps=1/3,
    sigma=0.1,
    budget=10,
    train_data_input=X,
    train_data_output=Y,
    arm_dictionary=arms_dict,
    trials_per_step=10
)

auto_model_generation3 = AlgorithmSelectionAdaptiveSRB(
    exp_param=1,
    eps=1/3,
    budget=15,
    train_data_input=X,
    train_data_output=Y,
    arm_dictionary=arms_dict,
    trials_per_step=10
)


if __name__ == "__main__":
    model = auto_model_generation3.learn()
    filename = base_dir + 'best_model.sav'
    pickle.dump(model, open(filename, 'wb'))
