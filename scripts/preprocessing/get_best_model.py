import optuna

loaded_study = optuna.load_study(study_name="hpo-ner-wrist-ct", storage="sqlite:///ner-wrist-ct.db")

with open("best_trial.txt", "a+") as f:
    params = loaded_study.best_params
    f.write(f"val_F1: {loaded_study.best_value}\n")
    for key in params.keys():
        f.write(f"{key}: {params[key]}\n")
        f.write("\n\n")

print(100 * "xv")
print(loaded_study.best_trial)

print(20 * "++++++++++++++++++++++++++")
print(loaded_study.best_trials)
print(100 * "+")
print(loaded_study.best_trial)
print(100 * "+")
print(loaded_study.best_params)
print(100 * "+")
print(loaded_study.best_value)
print(100 * "+")
print(loaded_study.trial_dataframe)
