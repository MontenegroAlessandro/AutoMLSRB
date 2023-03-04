# Libraries
import pickle
import numpy as np
from copy import deepcopy

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from Model.model_wrappers import MyMLPClassifier

from Datasets.ClsDatasets import SteelPlatesFaultDataset
from sklearn.preprocessing import MinMaxScaler

from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, Constant
from ConfigSpace.conditions import EqualsCondition

from HyperparametersOptimization.hyperparemeters_optimization import TunerGenetic
from AutomaticModelSelection.automatic_model_selection import Arm, EfficientCASHRB, AlgorithmSelectionSRB, \
    AlgorithmSelectionAdaptiveSRB, BaseAlgorithmSelection

from sklearnex import patch_sklearn

patch_sklearn()
# *********************************************************************************************************************

# Data keeping and preprocessing
data = SteelPlatesFaultDataset()
X = data.input
Y = np.zeros(len(data.target), dtype=int)
for i in range(len(data.target)):
    if data.target[i] == '1':
        Y[i] = 0
    elif data.target[i] == '2':
        Y[i] = 1
print(X.shape, Y.shape)
# data for the net need to be normalized
X_net = MinMaxScaler().fit_transform(data.input)
# *********************************************************************************************************************

# Hyperparameter(s)
# LogisticRegression
hp_dict_logistic_reg = dict(
    penalty=Constant("penalty", "l2"),
    tol=UniformFloatHyperparameter(name="tol", lower=1e-6, upper=1e-2, default_value=1e-4),
    C=UniformFloatHyperparameter(name="C", lower=0.01, upper=100, default_value=1),
    class_weight=Constant("class_weight", "balanced"),
    solver=CategoricalHyperparameter(name="solver",
                                     choices=["lbfgs", "newton-cg", "sag", "saga"],
                                     default_value="lbfgs"),
    max_iter=UniformIntegerHyperparameter(name="max_iter", lower=1000, upper=2000, default_value=1000)
)

# Support Vector Machines
hp_dict_svm = dict(
    C=UniformFloatHyperparameter(name="C", lower=0.01, upper=100, default_value=1),
    kernel=Constant("kernel", "rbf"),
    gamma=UniformFloatHyperparameter(name="gamma", lower=0.005, upper=0.1, default_value=0.03),
    tol=UniformFloatHyperparameter(name="tol", lower=1e-6, upper=1e-2, default_value=1e-4),
    class_weight=Constant("class_weight", "balanced")
)

# AdaBoost
hp_dict_adaboost = dict(
    n_estimators=UniformIntegerHyperparameter(name="n_estimators", lower=50, upper=500, default_value=200),
    learning_rate=UniformFloatHyperparameter(name="learning_rate", lower=0.001, upper=1, default_value=0.1),
    algorithm=CategoricalHyperparameter(name="algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")
)

# XGBoost
hp_dict_xgb = dict(
    n_estimators=UniformIntegerHyperparameter(name="n_estimators", lower=50, upper=500, default_value=200),
    eta=UniformFloatHyperparameter(name="eta", lower=0.01, upper=1, default_value=0.3),
    min_child_weight=UniformIntegerHyperparameter(name="min_child_weight", lower=1, upper=10, default_value=1),
    max_depth=UniformIntegerHyperparameter(name="max_depth", lower=4, upper=12, default_value=6),
    subsample=UniformFloatHyperparameter(name="subsample", lower=0.2, upper=1, default_value=0.5),
    gamma=UniformFloatHyperparameter(name="gamma", lower=0, upper=10, default_value=0),
    alpha=UniformFloatHyperparameter(name="alpha", lower=1e-10, upper=1, default_value=1e-10)
    # lambda_t=UniformFloatHyperparameter(name="lambda_t", lower=1e-10, upper=1, default_value=1e-10)
)

# RandomForest
hp_dict_rf = dict(
    n_estimators=UniformIntegerHyperparameter(name="n_estimators", lower=50, upper=500, default_value=50),
    criterion=CategoricalHyperparameter(name="criterion", choices=["gini", "entropy", "log_loss"],
                                        default_value="gini"),
    max_depth=UniformIntegerHyperparameter(name="max_depth", lower=4, upper=12, default_value=6),
    max_features=CategoricalHyperparameter(name="max_features", choices=["sqrt", "log2"], default_value="sqrt"),
    bootstrap=CategoricalHyperparameter(name="bootstrap", choices=[True], default_value=True),
    oob_score=CategoricalHyperparameter(name="oob_score", choices=[True], default_value=True),
    class_weight=Constant("class_weight", "balanced")
)

# Extremely Randomized Trees
hp_dict_extra_trees = deepcopy(hp_dict_rf)

# KNN
hp_dict_knn = dict(
    n_neighbors=UniformIntegerHyperparameter(name="n_neighbors", lower=10, upper=100, default_value=10),
    weights=CategoricalHyperparameter(name="weights", choices=["uniform", "distance"], default_value="uniform"),
    algorithm=CategoricalHyperparameter(name="algorithm", choices=["ball_tree", "kd_tree"],
                                        default_value="kd_tree"),
    leaf_size=UniformIntegerHyperparameter(name="leaf_size", lower=10, upper=50, default_value=30),
    p=CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)
)

# MultiLayerPerceptron
hp_dict_mlp = dict(
    hidden_layer_number=UniformIntegerHyperparameter(name="hidden_layer_number", lower=1, upper=5, default_value=1),
    hidden_layer_size=UniformIntegerHyperparameter(name="hidden_layer_size", lower=10, upper=100, default_value=10),
    activation=CategoricalHyperparameter(name="activation", choices=["tanh", "relu"],
                                         default_value="relu"),
    solver=Constant("solver", "adam"),
    alpha=UniformFloatHyperparameter(name="alpha", lower=1e-7, upper=1., default_value=0.0001),
    learning_rate=Constant("learning_rate", "adaptive"),
    learning_rate_init=UniformFloatHyperparameter(name="learning_rate_init", lower=1e-4, upper=1,
                                                  default_value=0.001),
    tol=UniformFloatHyperparameter(name="tol", lower=1e-5, upper=1e-2, default_value=1e-4),
    momentum=UniformFloatHyperparameter(name="momentum", lower=0.6, upper=1, q=0.05, default_value=0.9),
    beta_1=UniformFloatHyperparameter(name="beta_1", lower=0.6, upper=1, default_value=0.9),
    max_iter=UniformIntegerHyperparameter(name="max_iter", lower=1000, upper=2000, default_value=1000)
)

# SMBO
hps = [hp_dict_logistic_reg, hp_dict_svm, hp_dict_knn, hp_dict_adaboost, hp_dict_xgb, hp_dict_rf, hp_dict_extra_trees,
       hp_dict_mlp]
hp_dict_smbo = {}
for elem in hps:
    for key in elem:
        hp_dict_smbo[key] = elem[key]
values = np.arange(8).tolist()
hp_dict_smbo["root"] = CategoricalHyperparameter(name="root", choices=values, default_value=0)


# *********************************************************************************************************************

# Objective(s)
def objective_logistic_reg(config):
    model = LogisticRegression(
        penalty=config["penalty"],
        tol=config["tol"],
        C=config["C"],
        class_weight=config["class_weight"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    try:
        scores = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring="accuracy")
    except (ValueError, AttributeError):
        scores = np.zeros(1)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()


def objective_svm(config):
    model = SVC(
        C=config["C"],
        kernel=config["kernel"],
        gamma=config["gamma"],
        tol=config["tol"],
        class_weight=config["class_weight"]
    )
    try:
        scores = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring="accuracy")
    except (ValueError, AttributeError):
        scores = np.zeros(1)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()


def objective_adaboost(config):
    model = AdaBoostClassifier(
        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        algorithm=config["algorithm"]
    )
    try:
        scores = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring="accuracy")
    except (ValueError, AttributeError):
        scores = np.zeros(1)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()


def objective_xgboost(config):
    model = xgb.XGBClassifier(
        n_estimators=config["n_estimators"],
        eta=config["eta"],
        min_child_weight=config["min_child_weight"],
        max_depth=config["max_depth"],
        subsample=config["subsample"],
        gamma=config["gamma"],
        alpha=config["alpha"]
        # lambda_t=config["lambda_t"]
    )
    try:
        scores = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring="accuracy")
    except (ValueError, AttributeError):
        scores = np.zeros(1)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()


def objective_rf(config):
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        criterion=config["criterion"],
        max_depth=config["max_depth"],
        max_features=config["max_features"],
        bootstrap=config["bootstrap"],
        oob_score=config["oob_score"],
        class_weight=config["class_weight"]
    )
    try:
        scores = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring="accuracy")
    except (ValueError, AttributeError):
        scores = np.zeros(1)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()


def objective_extra_trees(config):
    model = ExtraTreesClassifier(
        n_estimators=config["n_estimators"],
        criterion=config["criterion"],
        max_depth=config["max_depth"],
        max_features=config["max_features"],
        bootstrap=config["bootstrap"],
        oob_score=config["oob_score"],
        class_weight=config["class_weight"]
    )
    try:
        scores = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring="accuracy")
    except (ValueError, AttributeError):
        scores = np.zeros(1)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()


def objective_knn(config):
    model = KNeighborsClassifier(
        n_neighbors=config["n_neighbors"],
        weights=config["weights"],
        algorithm=config["algorithm"],
        leaf_size=config["leaf_size"],
        p=config["p"]
    )
    try:
        scores = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring="accuracy")
    except (ValueError, AttributeError):
        scores = np.zeros(1)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()


def objective_mlp(config):
    my_model = MyMLPClassifier(
        hidden_layer_size=config["hidden_layer_size"],
        hidden_layer_number=config["hidden_layer_number"],
        activation=config["activation"],
        solver=config["solver"],
        alpha=config["alpha"],
        learning_rate=config["learning_rate"],
        learning_rate_init=config["learning_rate_init"],
        max_iter=config["max_iter"],
        tol=config["tol"],
        momentum=config["momentum"],
        beta_1=config["beta_1"]
    )
    try:
        scores = cross_val_score(my_model, X_net, Y, cv=10, n_jobs=-1, scoring="accuracy")
    except (ValueError, AttributeError):
        scores = np.zeros(1)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()


def objective_smbo(config):
    models = [
        LogisticRegression(penalty=config["penalty"], tol=config["tol"], C=config["C"],
                           class_weight=config["class_weight"], solver=config["solver"], max_iter=config["max_iter"]),
        SVC(C=config["C"], kernel=config["kernel"], gamma=config["gamma"], tol=config["tol"],
            class_weight=config["class_weight"]),
        KNeighborsClassifier(n_neighbors=config["n_neighbors"], weights=config["weights"], p=config["p"],
                             algorithm=config["algorithm"], leaf_size=config["leaf_size"]),
        AdaBoostClassifier(n_estimators=config["n_estimators"], learning_rate=config["learning_rate"],
                           algorithm=config["algorithm"]),
        xgb.XGBClassifier(n_estimators=config["n_estimators"], eta=config["eta"],
                          min_child_weight=config["min_child_weight"], max_depth=config["max_depth"],
                          subsample=config["subsample"], gamma=config["gamma"], alpha=config["alpha"]),
        RandomForestClassifier(max_depth=config["max_depth"], criterion=config["criterion"],
                               n_estimators=config["n_estimators"], max_features=config["max_features"],
                               bootstrap=config["bootstrap"], oob_score=config["oob_score"],
                               class_weight=config["class_weight"]),
        ExtraTreesClassifier(max_depth=config["max_depth"], criterion=config["criterion"],
                             n_estimators=config["n_estimators"], max_features=config["max_features"],
                             bootstrap=config["bootstrap"], oob_score=config["oob_score"],
                             class_weight=config["class_weight"]),
        MyMLPClassifier(hidden_layer_size=config["hidden_layer_size"],
                        hidden_layer_number=config["hidden_layer_number"],
                        activation=config["activation"], solver=config["solver"],
                        alpha=config["alpha"], learning_rate=config["learning_rate"],
                        learning_rate_init=config["learning_rate_init"],
                        max_iter=config["max_iter"], tol=config["tol"], momentum=config["momentum"],
                        beta_1=config["beta_1"])
    ]
    model = models[config["root"]]
    print("ALGO: ", config["root"])
    try:
        if config["root"] == 7:
            scores = cross_val_score(model, X_net, Y, cv=10, n_jobs=-1, scoring="accuracy")
        else:
            scores = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring="accuracy")
    except (ValueError, AttributeError):
        scores = np.zeros(1)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()
# *********************************************************************************************************************

# Tuner(s)
tuner_args = dict(
    n_agents=10,
    n_generations=100,
    prob_point_mutation=0.5,
    tuning_mode="best_performant_elitism",
    pool_size=None,
    objective=objective_logistic_reg,
    hp_dict=deepcopy(hp_dict_logistic_reg),
    seed=2023,
    n_jobs=-1,
    which_res="cost",
    log_path="experiments/test_genetic/test_log_reg",
    conditions=None
)
tuner_gen_lr = TunerGenetic(**tuner_args)

tuner_args["objective"] = objective_svm
tuner_args["hp_dict"] = deepcopy(hp_dict_svm)
tuner_args["log_path"] = "experiments/test_genetic/test_svm"
tuner_gen_svm = TunerGenetic(**tuner_args)

tuner_args["objective"] = objective_knn
tuner_args["hp_dict"] = deepcopy(hp_dict_knn)
tuner_args["log_path"] = "experiments/test_genetic/test_knn"
tuner_gen_knn = TunerGenetic(**tuner_args)

tuner_args["objective"] = objective_adaboost
tuner_args["hp_dict"] = deepcopy(hp_dict_adaboost)
tuner_args["log_path"] = "experiments/test_genetic/test_ada"
tuner_gen_ada = TunerGenetic(**tuner_args)

tuner_args["objective"] = objective_xgboost
tuner_args["hp_dict"] = deepcopy(hp_dict_xgb)
tuner_args["log_path"] = "experiments/test_genetic/test_xgb"
tuner_gen_xgb = TunerGenetic(**tuner_args)

tuner_args["objective"] = objective_rf
tuner_args["hp_dict"] = deepcopy(hp_dict_rf)
tuner_args["log_path"] = "experiments/test_genetic/test_rf"
tuner_gen_rf = TunerGenetic(**tuner_args)

tuner_args["objective"] = objective_extra_trees
tuner_args["hp_dict"] = deepcopy(hp_dict_extra_trees)
tuner_args["log_path"] = "experiments/test_genetic/test_et"
tuner_gen_et = TunerGenetic(**tuner_args)

tuner_args["objective"] = objective_mlp
tuner_args["hp_dict"] = deepcopy(hp_dict_mlp)
tuner_args["log_path"] = "experiments/test_genetic/test_mlp"
tuner_gen_mlp = TunerGenetic(**tuner_args)

conditions = []
for k in hp_dict_logistic_reg:
    conditions.append(EqualsCondition(hp_dict_smbo[k], hp_dict_smbo["root"], 0))
for k in hp_dict_svm:
    conditions.append(EqualsCondition(hp_dict_smbo[k], hp_dict_smbo["root"], 1))
for k in hp_dict_knn:
    conditions.append(EqualsCondition(hp_dict_smbo[k], hp_dict_smbo["root"], 2))
for k in hp_dict_adaboost:
    conditions.append(EqualsCondition(hp_dict_smbo[k], hp_dict_smbo["root"], 3))
for k in hp_dict_xgb:
    conditions.append(EqualsCondition(hp_dict_smbo[k], hp_dict_smbo["root"], 4))
for k in hp_dict_rf:
    conditions.append(EqualsCondition(hp_dict_smbo[k], hp_dict_smbo["root"], 5))
for k in hp_dict_extra_trees:
    conditions.append(EqualsCondition(hp_dict_smbo[k], hp_dict_smbo["root"], 6))
for k in hp_dict_mlp:
    conditions.append(EqualsCondition(hp_dict_smbo[k], hp_dict_smbo["root"], 7))
tuner_args["objective"] = objective_smbo
tuner_args["hp_dict"] = deepcopy(hp_dict_smbo)
tuner_args["log_path"] = "experiments/test_genetic/test_smbo"
tuner_args["conditions"] = conditions
tuner_gen_smbo = TunerGenetic(**tuner_args)
# *********************************************************************************************************************

# Arm(s)
arm_logistic_reg = Arm(model=LogisticRegression, tuner=tuner_gen_lr)
arm_svm = Arm(model=SVC, tuner=tuner_gen_svm)
arm_adaboost = Arm(model=AdaBoostClassifier, tuner=tuner_gen_ada)
arm_xgboost = Arm(model=xgb.XGBClassifier, tuner=tuner_gen_xgb)
arm_rf = Arm(model=RandomForestClassifier, tuner=tuner_gen_rf)
arm_extra_trees = Arm(model=ExtraTreesClassifier, tuner=tuner_gen_et)
arm_knn = Arm(model=KNeighborsClassifier, tuner=tuner_gen_knn)
arm_mlp = Arm(model=MyMLPClassifier, tuner=tuner_gen_mlp)

# Dictionary of Arm(s)
arms_dict = dict(
    logistic_reg=arm_logistic_reg,
    svm=arm_svm,
    knn=arm_knn,
    adaboost=arm_adaboost,
    xgboost=arm_xgboost,
    random_forest=arm_rf,
    extra_trees=arm_extra_trees,
    mlp=arm_mlp
)
# *********************************************************************************************************************

for key in arms_dict:
    arms_dict[key].tuner.tune()

tuner_gen_smbo.tune()
