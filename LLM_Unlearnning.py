from unsloth import FastLanguageModel
import torch
max_seq_length = 500 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn  import pad_sequence # Corrected import
import datasets
# # from utils import add_dataset_index, get_model_identifiers_from_yaml
# from data_module import convert_raw_data_to_model_format
import os
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
import transformers
from transformers import Trainer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

from trl import SFTTrainer
from data_preparation.data_prep import data_preparation_full, data_preparation_retain, data_preparation_forget
from optimization.optimization_functions import GradientAscentSFTTrainer, ScaledGradientAscentTrainer, WeightedUnlearningTrainer
from datasets import load_dataset


def data_preparation_full_():
    alpaca_prompt = """Answer the following question:
### Question:
{}

### Answer:
{}"""
    
    def formatting_prompts_func(examples):
        texts = [alpaca_prompt.format(question, answer) for question, answer in zip(examples["question"], examples["answer"])]
        return {"text": texts}
    
    # Load the dataset
    print("Loading TOFU dataset...")
    dataset = load_dataset("locuslab/TOFU", "full", split="train")
    print(f"Raw dataset: {dataset}")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(f"Formatted dataset: {dataset}")
    return dataset

class LLM_Unlearnning:
    def __init__(self,model_name, max_seq_length, load_in_4bit):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,  # or choose "unsloth/Llama-3.2-1B-Instruct"
            max_seq_length=self.max_seq_length,
            dtype=dtype,
            load_in_4bit=self.load_in_4bit,
        )

    # Loading the Peft (Lora) model is used and full dataset is used for training
    def train(self):
        self.model_full_dataset = FastLanguageModel.get_peft_model(
            self.base_model,
            r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", ],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=True,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )

        trainer = SFTTrainer(
            model=self.model_full_dataset,
            tokenizer=self.tokenizer,
            train_dataset=data_preparation_full(),
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences.
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                # num_train_epochs = 1, # Set this for 1 full training run.
                max_steps=60,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="none",  # Use this for WandB etc
            ),
        )

        trainer.train()

    # Loading the Peft (Lora) model is used and rerain dataset is used for training
    def train_with_retain_dataset(self):
        self.retain_model = FastLanguageModel.get_peft_model(
            self.base_model,
            r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", ],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=True,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )

        trainer = SFTTrainer(
            model=self.retain_model,
            tokenizer=self.tokenizer,
            train_dataset=data_preparation_retain(),
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences.
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                # num_train_epochs = 1, # Set this for 1 full training run.
                max_steps=60,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="none",  # Use this for WandB etc
            ),
        )

        trainer.train()

    # Loading the Peft (Lora) model is used and forget dataset is used for unlearning
    def finetune_with_forget_dataset(self):
        self.forget_model = FastLanguageModel.get_peft_model(
            self.model_full_dataset,
            r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", ],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=True,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )

        trainer = GradientAscentSFTTrainer(
            model=self.forget_model,
            tokenizer=self.tokenizer,
            train_dataset=data_preparation_forget(),
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences.
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                # num_train_epochs = 1, # Set this for 1 full training run.
                max_steps=60,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="none",  # Use this for WandB etc
            ),
        )

        trainer.train()




llm_unlearnning = LLM_Unlearnning("unsloth/llama-3-8b-bnb-4bit", max_seq_length=500, load_in_4bit=True)

# Finetuning using full datatset
llm_unlearnning.train()

# Finetuning using retain datatset
llm_unlearnning.train_with_retain_dataset()

# Finetuning using forget datatset
llm_unlearnning.finetune_with_forget_dataset()
