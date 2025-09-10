import os
import pandas as pd
import torch
import wandb
from sklearn.model_selection import KFold
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import DataCollatorForCompletionOnlyLM
from transformers.utils import is_torch_bf16_gpu_available

import constants
import utils

def main():
    os.makedirs(constants.OUTPUT_DIR, exist_ok=True)

    if constants.WANDB:
        wandb.init(project=constants.COMPETITION_NAME, name=constants.EXP_NAME)

    # 1. Load and preprocess data
    df = pd.read_csv(constants.DATA_PATH)
    if constants.IS_DEBUG:
        df = df.sample(50, random_state=constants.SEED).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME)
    tokenizer.padding_side = "left"

    df = utils.create_prompts(df, tokenizer)
    df = utils.preprocess_df(df, tokenizer)

    # 2. CV Split
    kf = KFold(n_splits=constants.N_FOLDS, shuffle=True, random_state=constants.SEED)
    train_idx, val_idx = list(kf.split(df))[constants.FOLD]
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]

    train_dataset = utils.ClassifyDataset(df_train)
    val_dataset = utils.ClassifyDataset(df_val)

    # 3. Model Setup
    model = AutoModelForCausalLM.from_pretrained(
        constants.MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True
    )
    
    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM, bias='none',
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Trainer Setup
    training_args = TrainingArguments(
        output_dir=constants.MODEL_OUTPUT_PATH,
        num_train_epochs=constants.EPOCH,
        learning_rate=constants.LR,
        per_device_train_batch_size=constants.TRAIN_BS,
        per_device_eval_batch_size=constants.EVAL_BS,
        gradient_accumulation_steps=constants.GRAD_ACC_NUM,
        optim="paged_adamw_8bit",
        logging_steps=10,
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=10,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=is_torch_bf16_gpu_available(),
        fp16=not is_torch_bf16_gpu_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=constants.SEED,
        report_to="wandb" if constants.WANDB else "none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForCompletionOnlyLM("Answer:", tokenizer=tokenizer),
    )

    # 5. Train
    trainer.train()
    trainer.save_model(constants.MODEL_OUTPUT_PATH)

if __name__ == "__main__":
    main()