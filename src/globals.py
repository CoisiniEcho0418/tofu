from pathlib import Path

import yaml

# with open("/home/wxy/wxy_workspace/LLM_unlearn/AlphaEdit-main/globals.yml", "r") as stream:
#     data = yaml.safe_load(stream)
with open("/home/wxy/wxy_workspace/LLM_unlearn/tofu/config/globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]

# ---
#   # Result files
#   RESULTS_DIR: "AlphaEdit-main/results"

#   # Data files
#   DATA_DIR: "AlphaEdit-main/data"
#   STATS_DIR: "AlphaEdit-main/data/stats"
#   KV_DIR: "AlphaEdit-main/share/projects/rewriting-knowledge/kvs"

#   # Hyperparameters
#   HPARAMS_DIR: "AlphaEdit-main/hparams"

#   # Remote URLs
#   REMOTE_ROOT_URL: "https://memit.baulab.info"
