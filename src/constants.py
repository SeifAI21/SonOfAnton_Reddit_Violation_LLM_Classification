import torch
from transformers.utils import is_torch_bf16_gpu_available

# Main configuration parameters
WANDB = False  # Enable/disable Weights & Biases logging
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
IS_DEBUG = True
N_FOLDS = 5
EPOCH = 1
LR = 1e-4
TRAIN_BS = 8
GRAD_ACC_NUM = 1
EVAL_BS = 8
FOLD = 0
SEED = 42


# Paths
EXP_ID = "jigsaw-lora-finetune-baseline"
if IS_DEBUG:
    EXP_ID += "_debug"
EXP_NAME = EXP_ID + f"_fold{FOLD}"
COMPETITION_NAME = "jigsaw-kaggle"
OUTPUT_DIR = f"./output/{EXP_NAME}/"
MODEL_OUTPUT_PATH = f"{OUTPUT_DIR}/trained_model"
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/train.csv"

# System Prompt
SYS_PROMPT = "You are an expert AI assistant tasked with content moderation. Your goal is to determine if a Reddit comment violates a specific community rule. You must analyze the provided examples carefully to understand the context of the rule. Your final response should be strictly 'Yes' if the comment violates the rule, or 'No' if it does not. Do not provide any other text, explanation, or reasoning."