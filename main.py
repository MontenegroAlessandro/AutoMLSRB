# Libraries
import pickle
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from Datasets.ClsDatasets import WineDataset
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from HyperparametersOptimization.hyperparemeters_optimization import TunerSMAC
from sklearn.model_selection import cross_val_score
from AutomaticModelGeneration.automatic_model_generation import Arm, BaseAlgorithmSelection, AlgorithmSelectionSRB

##### Data #####
data = WineDataset()
X = data.input
Y = data.target
print(X.shape, Y.shape)

##### Hyperparameter(s) #####
hp_dict_adaboost = dict(
    n_estimators=UniformIntegerHyperparameter("n_estimators", 1, 99),
    learning_rate=UniformFloatHyperparameter("learning_rate", 0, 10)
)

hp_dict_rf = dict(
    max_depth=UniformIntegerHyperparameter("max_depth", 1, 99)
)


##### Objective(s) #####
def objective_adaboost(config):
    model = AdaBoostClassifier(n_estimators=config["n_estimators"], learning_rate=config["learning_rate"])
    scores = cross_val_score(model, X, Y, cv=10)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()


def objective_rf(config):
    model = RandomForestClassifier(max_depth=config["max_depth"])
    scores = cross_val_score(model, X, Y, cv=10)
    print(scores.mean(), scores.std())
    return 1 - scores.mean()


##### Tuner(s) #####
base_dir = "experiments/Test1/"
tuner_args = dict(
    hp_dict=hp_dict_adaboost,
    objective_foo=objective_adaboost,
    trials=10,
    log_path=base_dir + "test_ada",
    n_jobs=1,
    seed=2023
)
tuner_adaboost = TunerSMAC(**tuner_args)

tuner_args["hp_dict"] = hp_dict_rf
tuner_args["objective_foo"] = objective_rf
tuner_args["log_path"] = base_dir + "test_rf"
tuner_rf = TunerSMAC(**tuner_args)

##### Arm(s) #####
arm_adaboost = Arm(model=AdaBoostClassifier, tuner=tuner_adaboost)
arm_rf = Arm(model=RandomForestClassifier, tuner=tuner_rf)

##### Dictionary of Arm(s) #####
arms_dict = dict(
    adaboost=arm_adaboost,
    random_forest=arm_rf
)

##### Automatic Block ######
auto_model_generation = BaseAlgorithmSelection(
    budget=10,
    train_data_input=X,
    train_data_output=Y,
    arm_dictionary=arms_dict,
    trials_per_step=10
)

auto_model_generation2 = AlgorithmSelectionSRB(
    exp_param=1,
    eps=0.25,
    sigma=0.1,
    budget=10,
    train_data_input=X,
    train_data_output=Y,
    arm_dictionary=arms_dict,
    trials_per_step=10
)

if __name__ == "__main__":
    model = auto_model_generation2 .learn()
    filename = base_dir + 'best_model.sav'
    pickle.dump(model, open(filename, 'wb'))
