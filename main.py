import neptune
import secret_config
run = neptune.init_run(
    project="secret_config.neptune_project",
    api_token="secret_config.neptune_api_token",
)  # your credentials, make sure you have the secret_config file and that it's in gitignore


params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

for epoch in range(10):
    run["train/loss"].append(0.9 ** epoch)
    # insert your training loop here

run["eval/f1_score"] = 0.66

run.stop()