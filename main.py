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

from HyperparametersOptimization.hyperparemeters_optimization import TunerSMAC, TunerBOHB
from AutomaticModelSelection.automatic_model_selection import Arm, EfficientCASHRB, AlgorithmSelectionSRB, \
    AlgorithmSelectionAdaptiveSRB, BaseAlgorithmSelection

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
    penalty=CategoricalHyperparameter(name="penalty", choices=["l1", "l2", "elasticnet"], default_value="l2"),
    tol=UniformFloatHyperparameter(name="tol", lower=1e-6, upper=1e-1, default_value=1e-4),
    C=UniformFloatHyperparameter(name="C", lower=0.03125, upper=10, default_value=1),
    class_weight=CategoricalHyperparameter(name="class_weight", choices=["balanced"], default_value=None),
    solver=CategoricalHyperparameter(name="solver",
                                     choices=["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                                     default_value="lbfgs"),
    max_iter=UniformIntegerHyperparameter(name="max_iter", lower=400, upper=1000, default_value=400)
)

# Support Vector Machines
hp_dict_svm = dict(
    C=UniformFloatHyperparameter(name="C", lower=0.03125, upper=10, default_value=1),
    kernel=CategoricalHyperparameter(name="kernel", choices=["rbf", "linear", "poly", "sigmoid"], default_value="rbf"),
    degree=UniformIntegerHyperparameter(name="degree", lower=1, upper=5, default_value=3),
    gamma=CategoricalHyperparameter(name="gamma", choices=["scale", "auto"], default_value="auto"),
    coef0=UniformFloatHyperparameter(name="coef0", lower=0, upper=3, default_value=0),
    probability=CategoricalHyperparameter(name="probability", choices=[True, False], default_value=True),
    tol=UniformFloatHyperparameter(name="tol", lower=1e-6, upper=1e-1, default_value=1e-4),
    class_weight=CategoricalHyperparameter(name="class_weight", choices=["balanced"], default_value=None),
    decision_function_shape=CategoricalHyperparameter(name="decision_function_shape", choices=["ovr", "ovo"],
                                                      default_value="ovr")
)

# AdaBoost
hp_dict_adaboost = dict(
    n_estimators=UniformIntegerHyperparameter(name="n_estimators", lower=50, upper=500, default_value=200),
    learning_rate=UniformFloatHyperparameter(name="learning_rate", lower=0.01, upper=3, default_value=0.1),
    algorithm=CategoricalHyperparameter(name="algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")
)

# XGBoost
hp_dict_xgb = dict(
    n_estimators=UniformIntegerHyperparameter(name="n_estimators", lower=50, upper=500, default_value=200),
    eta=UniformFloatHyperparameter(name="eta", lower=0.025, upper=0.3, default_value=0.3),
    min_child_weight=UniformIntegerHyperparameter(name="min_child_weight", lower=1, upper=10, default_value=1),
    max_depth=UniformIntegerHyperparameter(name="max_depth", lower=2, upper=10, default_value=6),
    subsample=UniformFloatHyperparameter(name="subsample", lower=0.5, upper=1, default_value=1),
    gamma=UniformFloatHyperparameter(name="gamma", lower=0, upper=1, default_value=0),
    colsample_bytree=UniformFloatHyperparameter(name="colsample_bytree", lower=0.5, upper=1, default_value=1.),
    alpha=UniformFloatHyperparameter(name="alpha", lower=1e-10, upper=10, default_value=1e-10),
    lambda_t=UniformFloatHyperparameter(name="lambda_t", lower=1e-10, upper=10, default_value=1e-10),
    scale_pos_weight=CategoricalHyperparameter(name="scale_pos_weight", choices=[0.01, 0.1, 1., 10, 100],
                                               default_value=1.)
)

# RandomForest
hp_dict_rf = dict(
    n_estimators=UniformIntegerHyperparameter(name="n_estimators", lower=50, upper=500, default_value=50),
    criterion=CategoricalHyperparameter(name="criterion", choices=["gini", "entropy", "log_loss"],
                                        default_value="gini"),
    max_depth=UniformIntegerHyperparameter(name="max_depth", lower=1, upper=200, default_value=30),
    min_samples_split=UniformIntegerHyperparameter(name="min_samples_split", lower=2, upper=10, default_value=2),
    min_samples_leaf=UniformIntegerHyperparameter(name="min_samples_leaf", lower=2, upper=5, default_value=2),
    max_features=CategoricalHyperparameter(name="max_features", choices=["sqrt", "log2"], default_value="sqrt"),
    bootstrap=CategoricalHyperparameter(name="bootstrap", choices=[True, False], default_value=False),
    oob_score=CategoricalHyperparameter(name="oob_score", choices=[True, False], default_value=False),
    class_weight=CategoricalHyperparameter(name="class_weight", choices=["balanced", "balanced_subsample"],
                                           default_value=None),
    ccp_alpha=UniformFloatHyperparameter(name="ccp_alpha", lower=0.0, upper=3, default_value=0.0)
)

# Extremely Randomized Trees
hp_dict_extra_trees = deepcopy(hp_dict_rf)

# KNN
hp_dict_knn = dict(
    n_neighbors=UniformIntegerHyperparameter(name="n_neighbors", lower=1, upper=100, default_value=1),
    weights=CategoricalHyperparameter(name="weights", choices=["uniform", "distance"], default_value="uniform"),
    algorithm=CategoricalHyperparameter(name="algorithm", choices=["auto", "ball_tree", "kd_tree", "brute"],
                                        default_value="auto"),
    leaf_size=UniformIntegerHyperparameter(name="leaf_size", lower=10, upper=50, default_value=30),
    p=UniformIntegerHyperparameter(name="p", lower=1, upper=5, default_value=2)
)

# MultiLayerPerceptron
hp_dict_mlp = dict(
    hidden_layer_number=UniformIntegerHyperparameter(name="hidden_layer_number", lower=1, upper=5, default_value=1),
    hidden_layer_size=UniformIntegerHyperparameter(name="hidden_layer_size", lower=10, upper=100, default_value=10),
    activation=CategoricalHyperparameter(name="activation", choices=["identity", "logistic", "tanh", "relu"],
                                         default_value="relu"),
    solver=Constant("solver", "adam"),
    alpha=UniformFloatHyperparameter(name="alpha", lower=1e-7, upper=1., default_value=0.0001),
    learning_rate=CategoricalHyperparameter(name="learning_rate", choices=["adaptive", "invscaling", "constant"],
                                            default_value="constant"),
    learning_rate_init=UniformFloatHyperparameter(name="learning_rate_init", lower=1e-4, upper=3e-1,
                                                  default_value=0.001),
    tol=UniformFloatHyperparameter(name="tol", lower=1e-5, upper=1e-2, default_value=1e-4),
    momentum=UniformFloatHyperparameter(name="momentum", lower=0.6, upper=1, q=0.05, default_value=0.9),
    beta_1=UniformFloatHyperparameter(name="beta_1", lower=0.6, upper=1, default_value=0.9),
    power_t=UniformFloatHyperparameter(name="power_t", lower=1e-5, upper=1, default_value=0.5),
    max_iter=UniformIntegerHyperparameter(name="max_iter", lower=200, upper=1000, default_value=200)
)

# SMBO
hps = [hp_dict_adaboost, hp_dict_rf, hp_dict_knn, hp_dict_mlp]
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
        scores = cross_val_score(model, X, Y, cv=10, n_jobs=-1)
    except (ValueError, AttributeError):
        scores = np.zeros(1)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()


def objective_svm(config):
    model = SVC(
        C=config["C"],
        kernel=config["kernel"],
        degree=config["degree"],
        gamma=config["gamma"],
        coef0=config["coef0"],
        probability=config["probability"],
        tol=config["tol"],
        class_weight=config["class_weight"],
        decision_function_shape=config["decision_function_shape"]
    )
    try:
        scores = cross_val_score(model, X, Y, cv=10, n_jobs=-1)
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
        scores = cross_val_score(model, X, Y, cv=10, n_jobs=-1)
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
        colsample_bytree=config["colsample_bytree"],
        alpha=config["alpha"],
        lambda_t=config["lambda_t"],
        scale_pos_weight=config["scale_pos_weight"]
    )
    try:
        scores = cross_val_score(model, X, Y, cv=10, n_jobs=-1)
    except (ValueError, AttributeError):
        scores = np.zeros(1)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()


def objective_rf(config):
    model = RandomForestClassifier(
        max_depth=config["max_depth"],
        criterion=config["criterion"],
        n_estimators=config["n_estimators"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        bootstrap=config["bootstrap"],
        oob_score=config["oob_score"],
        class_weight=config["class_weight"],
        ccp_alpha=config["ccp_alpha"]
    )
    try:
        scores = cross_val_score(model, X, Y, cv=10, n_jobs=-1)
    except (ValueError, AttributeError):
        scores = np.zeros(1)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()


def objective_extra_trees(config):
    model = ExtraTreesClassifier(
        max_depth=config["max_depth"],
        criterion=config["criterion"],
        n_estimators=config["n_estimators"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        bootstrap=config["bootstrap"],
        oob_score=config["oob_score"],
        class_weight=config["class_weight"],
        ccp_alpha=config["ccp_alpha"]
    )
    try:
        scores = cross_val_score(model, X, Y, cv=10, n_jobs=-1)
    except (ValueError, AttributeError):
        scores = np.zeros(1)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()


def objective_knn(config):
    model = KNeighborsClassifier(
        n_neighbors=config["n_neighbors"],
        weights=config["weights"],
        p=config["p"],
        algorithm=config["algorithm"],
        leaf_size=config["leaf_size"]
    )
    try:
        scores = cross_val_score(model, X, Y, cv=10, n_jobs=-1)
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
        power_t=config["power_t"],
        max_iter=config["max_iter"],
        tol=config["tol"],
        momentum=config["momentum"],
        beta_1=config["beta_1"]
    )
    try:
        scores = cross_val_score(my_model, X_net, Y, cv=10, n_jobs=-1)
    except (ValueError, AttributeError):
        scores = np.zeros(1)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()


def objective_smbo(config):
    models = [
        LogisticRegression(penalty=config["penalty"], tol=config["tol"], C=config["C"],
                           class_weight=config["class_weight"], solver=config["solver"], max_iter=config["max_iter"]),
        SVC(C=config["C"], kernel=config["kernel"], degree=config["degree"], gamma=config["gamma"],
            coef0=config["coef0"], probability=config["probability"], tol=config["tol"],
            class_weight=config["class_weight"], decision_function_shape=config["decision_function_shape"]),
        KNeighborsClassifier(n_neighbors=config["n_neighbors"], weights=config["weights"], p=config["p"],
                             algorithm=config["algorithm"], leaf_size=config["leaf_size"]),
        AdaBoostClassifier(n_estimators=config["n_estimators"], learning_rate=config["learning_rate"],
                           algorithm=config["algorithm"]),
        xgb.XGBClassifier(n_estimators=config["n_estimators"], eta=config["eta"],
                          min_child_weight=config["min_child_weight"], max_depth=config["max_depth"],
                          subsample=config["subsample"], gamma=config["gamma"],
                          colsample_bytree=config["colsample_bytree"], alpha=config["alpha"],
                          lambda_t=config["lambda_t"], scale_pos_weight=config["scale_pos_weight"]),
        RandomForestClassifier(max_depth=config["max_depth"], criterion=config["criterion"],
                               n_estimators=config["n_estimators"], min_samples_split=config["min_samples_split"],
                               min_samples_leaf=config["min_samples_leaf"], max_features=config["max_features"],
                               bootstrap=config["bootstrap"], oob_score=config["oob_score"],
                               class_weight=config["class_weight"], ccp_alpha=config["ccp_alpha"]),
        ExtraTreesClassifier(max_depth=config["max_depth"], criterion=config["criterion"],
                             n_estimators=config["n_estimators"], min_samples_split=config["min_samples_split"],
                             min_samples_leaf=config["min_samples_leaf"], max_features=config["max_features"],
                             bootstrap=config["bootstrap"], oob_score=config["oob_score"],
                             class_weight=config["class_weight"], ccp_alpha=config["ccp_alpha"]),
        MyMLPClassifier(hidden_layer_size=config["hidden_layer_size"],
                        hidden_layer_number=config["hidden_layer_number"],
                        activation=config["activation"], solver=config["solver"],
                        alpha=config["alpha"], learning_rate=config["learning_rate"],
                        learning_rate_init=config["learning_rate_init"], power_t=config["power_t"],
                        max_iter=config["max_iter"], tol=config["tol"], momentum=config["momentum"],
                        beta_1=config["beta_1"])
    ]
    model = models[config["root"]]
    print("ALGO: ", config["root"])
    try:
        if config["root"] == 7:
            scores = cross_val_score(model, X_net, Y, cv=10, n_jobs=-1)
        else:
            scores = cross_val_score(model, X, Y, cv=10, n_jobs=-1)
    except (ValueError, AttributeError):
        scores = np.zeros(1)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()
# *********************************************************************************************************************

# Tuner(s)
base_dir = "experiments/Test_Exp1/"
tuner_args = dict(
    hp_dict=deepcopy(hp_dict_adaboost),
    objective_foo=objective_adaboost,
    trials=1,
    log_path=base_dir + "test_ada",
    n_jobs=1,
    seed=2023
)
tuner_adaboost = TunerSMAC(**tuner_args)

tuner_args["hp_dict"] = deepcopy(hp_dict_xgb)
tuner_args["objective_foo"] = objective_xgboost
tuner_args["log_path"] = base_dir + "test_xgb"
tuner_xgb = TunerSMAC(**tuner_args)

tuner_args["hp_dict"] = deepcopy(hp_dict_rf)
tuner_args["objective_foo"] = objective_rf
tuner_args["log_path"] = base_dir + "test_rf"
tuner_rf = TunerSMAC(**tuner_args)

tuner_args["hp_dict"] = deepcopy(hp_dict_extra_trees)
tuner_args["objective_foo"] = objective_extra_trees
tuner_args["log_path"] = base_dir + "test_extra_trees"
tuner_extra_trees = TunerSMAC(**tuner_args)

tuner_args["hp_dict"] = deepcopy(hp_dict_logistic_reg)
tuner_args["objective_foo"] = objective_logistic_reg
tuner_args["log_path"] = base_dir + "test_logistic"
tuner_logistic_reg = TunerSMAC(**tuner_args)

tuner_args["hp_dict"] = deepcopy(hp_dict_svm)
tuner_args["objective_foo"] = objective_svm
tuner_args["log_path"] = base_dir + "test_svm"
tuner_svm = TunerSMAC(**tuner_args)

tuner_args["hp_dict"] = deepcopy(hp_dict_knn)
tuner_args["objective_foo"] = objective_knn
tuner_args["log_path"] = base_dir + "test_knn"
tuner_knn = TunerSMAC(**tuner_args)

tuner_args["hp_dict"] = deepcopy(hp_dict_smbo)
tuner_args["objective_foo"] = objective_smbo
tuner_args["log_path"] = base_dir + "test_smbo"
tuner_smbo = TunerSMAC(**tuner_args)

tuner_args["hp_dict"] = deepcopy(hp_dict_mlp)
tuner_args["objective_foo"] = objective_mlp
tuner_args["log_path"] = base_dir + "test_mlp"

tuner_mlp = TunerSMAC(**tuner_args)
# *********************************************************************************************************************

# Arm(s)
arm_logistic_reg = Arm(model=LogisticRegression, tuner=tuner_logistic_reg)
arm_svm = Arm(model=SVC, tuner=tuner_svm)
arm_adaboost = Arm(model=AdaBoostClassifier, tuner=tuner_adaboost)
arm_xgboost = Arm(model=xgb.XGBClassifier, tuner=tuner_xgb)
arm_rf = Arm(model=RandomForestClassifier, tuner=tuner_rf)
arm_extra_trees = Arm(model=ExtraTreesClassifier, tuner=tuner_extra_trees)
arm_knn = Arm(model=KNeighborsClassifier, tuner=tuner_knn)
arm_mlp = Arm(model=MyMLPClassifier, tuner=tuner_mlp)

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

# Automatic Model Selection and Hyperparameters Optimization
budget = 300
comp_res = 2

# Base Round Robin
auto_model_generation = BaseAlgorithmSelection(
    budget=budget,
    train_data_input=X,
    train_data_output=Y,
    arm_dictionary=arms_dict,
    trials_per_step=comp_res,
    log_path=base_dir
)
model = auto_model_generation.learn()
filename = base_dir + 'best_model_base.sav'
pickle.dump(model, open(filename, 'wb'))

# Automatic Block Efficient CASH
auto_model_generation = EfficientCASHRB(
    budget=budget,
    train_data_input=X,
    train_data_output=Y,
    arm_dictionary=arms_dict,
    trials_per_step=comp_res,
    log_path=base_dir
)
model = auto_model_generation.learn()
filename = base_dir + 'best_model_ecash.sav'
pickle.dump(model, open(filename, 'wb'))

# Automatic Block SRB
auto_model_generation = AlgorithmSelectionSRB(
    budget=budget,
    train_data_input=X,
    train_data_output=Y,
    arm_dictionary=arms_dict,
    trials_per_step=comp_res,
    exp_param=1,
    eps=0.25,
    sigma=0.1,
    log_path=base_dir
)
model = auto_model_generation.learn()
filename = base_dir + 'best_model_rucb.sav'
pickle.dump(model, open(filename, 'wb'))

# Automatic Block Adaptive SRB
auto_model_generation = AlgorithmSelectionAdaptiveSRB(
    budget=budget,
    train_data_input=X,
    train_data_output=Y,
    arm_dictionary=arms_dict,
    trials_per_step=comp_res,
    exp_param=1,
    eps=0.25,
    log_path=base_dir
)
model = auto_model_generation.learn()
filename = base_dir + 'best_model_adarucb.sav'
pickle.dump(model, open(filename, 'wb'))

# SMBO
tuner_smbo.tune(budget*comp_res)
