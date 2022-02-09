import logging
import sys

import optuna

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "example-study"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
# study = optuna.create_study(study_name=study_name, storage=storage_name)
study = optuna.load_study(study_name=study_name, storage=storage_name)
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


study.optimize(objective, n_trials=3)