import os
import ray
from ray.rllib.models import ModelCatalog
from LLVMEnv.env.SynerEnv import LLVMEnv
from ray.rllib.algorithms import ppo
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from ray import tune
import random
import numpy as np
from ray import air
from ray.tune.registry import register_env
import argparse

def env_creator(env_config):
    return LLVMEnv(env_config)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, required=True, help="Path to the source file")
    args = parser.parse_args()

    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)

    ray.init(include_dashboard=True, ignore_reinit_error=True)

    env_name = 'CompilerGYM'
    env_config = {
        "source_file": f"{current_directory}/dataset/test/{args.filepath}",  # 使用命令行传入的路径
        "max_steps": 20,
        "reward_space": "IRInstCount",
        "reward_baseline": "IRInstCountOz",
        "llvm_version": "llvm-10.0.0",
        "llvm_tools_path": f"{current_directory}/llvm_tools/",
        "csv_path": f"{current_directory}/output/Phase2_Cluster_Synerpairs.csv",
    }

    register_env('CompilerGYM', lambda config: env_creator(env_config))

    ModelCatalog.register_custom_model("action_mask_model", TorchActionMaskModel)
    env = LLVMEnv(env_config)

    algo = ppo.PPOConfig() \
        .environment(env=env_name, disable_env_checking=True) \
        .framework("torch") \
        .training(
            model={
                "custom_model": "action_mask_model",
                "custom_model_config": {
                    "no_masking": False,
                },
            },
            _enable_learner_api=False,
            train_batch_size=128
        ) \
        .rollouts(num_rollout_workers=0, create_env_on_local_worker=True) \
        .rl_module(_enable_rl_module_api=False)

    algo["seed"] = 1234
    stop = {
        "episodes_total": 5000,
    }

    random.seed(1234)
    np.random.seed(1234)
    tuner = tune.Tuner(
        "PPO",
        param_space=algo,
        run_config=air.RunConfig(stop=stop),
    ).fit()
