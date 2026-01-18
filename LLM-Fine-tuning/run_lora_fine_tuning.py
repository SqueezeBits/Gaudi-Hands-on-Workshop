# This file is adapted from:
#   Optimum Habana - https://github.com/huggingface/optimum-habana
#   Licensed under the Apache License, Version 2.0
#
#   Modifications made by Yeonjoon Jung, 2025

import copy
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import transformers
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from peft import (
    LoraConfig,
    GraloraConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
)
from transformers.trainer_utils import is_main_process

from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments
from optimum.habana.utils import set_seed


try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:

    def check_optimum_habana_min_version(*a, **b):
        return ()


IGNORE_INDEX = -100

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

# Will error if the minimal version of Optimum Habana is not installed. Remove at your own risks.
check_optimum_habana_min_version("1.19.0.dev0")


@dataclass
class FineTuneArguments:
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen3-0.6B",
        metadata={"help": "The name or path to the model file."},
    )
    dataset_path: Optional[str] = field(
        default="./joseon_persona_dataset.csv", metadata={"help": "The path to the dataset file."}
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "Rank parameter in the LoRA method."},
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "Alpha parameter in the LoRA method."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout parameter in the LoRA method."},
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        metadata={"help": "Target modules for the LoRA/GraLoRA method."},
    )
    train_on_inputs: bool = field(
        default=True,
        metadata={"help": "if False, masks out inputs in loss"},
    )
    peft_type: str = field(
        default="lora",
        metadata={
            "help": ("The PEFT type to use."),
            "choices": ["lora", "gralora"],
        },
    )


def create_prompts(examples):
    prompts = {}
    prompts["source"] = []
    prompts["target"] = []
    for example in examples:
        # prompt_template = (
        #     "Below is an instruction that describes a task, paired with an input that provides further context. "
        #     "Write a response that appropriately completes the request.\n\n"
        #     "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        # )
        prompt_template = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{input}\n\n### Response:\n"
        )
        source = prompt_template.format_map(example)
        prompts["source"].append(source)
        prompts["target"].append(example["output"])
    return prompts


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((FineTuneArguments, GaudiTrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, padding_side="left")

    raw_datasets = load_dataset("csv", data_files=args.dataset_path)
    raw_datasets = raw_datasets.remove_columns(["instruction"])

    for key in raw_datasets:
        prompts = create_prompts(raw_datasets[key])
        columns_to_be_removed = list(raw_datasets[key].features.keys())
        raw_datasets[key] = raw_datasets[key].add_column("prompt_sources", prompts["source"])
        raw_datasets[key] = raw_datasets[key].add_column("prompt_targets", prompts["target"])
        raw_datasets[key] = raw_datasets[key].remove_columns(columns_to_be_removed)

    def tokenize(prompt, add_eos_token=True, add_bos_token=True):
        if hasattr(tokenizer, "add_eos_token"):
            add_eos_token_o = tokenizer.add_eos_token
        else:
            add_eos_token_o = None

        if hasattr(tokenizer, "add_bos_token"):
            add_bos_token_o = tokenizer.add_bos_token
        else:
            add_bos_token_o = None

        tokenizer.add_eos_token = add_eos_token
        tokenizer.add_bos_token = add_bos_token
        results = tokenizer(
            prompt,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors=None,
        )
        # restore original value
        if add_eos_token_o is not None:
            tokenizer.add_eos_token = add_eos_token_o

        if add_bos_token_o is not None:
            tokenizer.add_bos_token = add_bos_token_o

        for i in range(len(results["input_ids"])):
            if results["input_ids"][i][-1] != tokenizer.eos_token_id and add_eos_token:
                results["input_ids"][i].append(tokenizer.eos_token_id)
                results["attention_mask"][i].append(1)
        results["labels"] = copy.deepcopy(results["input_ids"])
        results["input_id_len"] = [len(result) for result in results["input_ids"]]
        return results

    def preprocess_function(examples):
        keys = list(examples.data.keys())
        if len(keys) != 2:
            raise ValueError(f"Unsupported dataset format, number of keys {keys} !=2")

        st = [s + t for s, t in zip(examples[keys[0]], examples[keys[1]])]
        add_bos_token = True
        examples_tokenized = tokenize(st, add_bos_token=add_bos_token)
        input_ids = examples_tokenized["input_ids"]
        labels = examples_tokenized["labels"]
        sources_tokenized = tokenize(examples[keys[0]], add_eos_token=False, add_bos_token=add_bos_token)
        for label, source_len in zip(labels, sources_tokenized["input_id_len"]):
            label[:source_len] = [IGNORE_INDEX] * source_len
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": examples_tokenized["attention_mask"],
        }

    with training_args.main_process_first(desc="dataset map pre-processing"):
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
        )

    train_dataset = tokenized_datasets["train"]

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, return_tensors="pt", mlm=False)
    logger.info("Using data collator of type {}".format(data_collator.__class__.__name__))

    # Load model
    config_kwargs = {
        "use_cache": False if training_args.gradient_checkpointing else True,
    }
    config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
    )

    # PEFT settings
    if args.peft_type == "lora":
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    elif args.peft_type == "gralora":
        peft_config = GraloraConfig(
            r=args.lora_rank,
            alpha=args.lora_alpha,
            gralora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    peft_model = get_peft_model(model, peft_config)
    peft_model.enable_input_require_grads()
    peft_model = peft_model.to(torch.bfloat16)

    peft_model.print_trainable_parameters()
    gaudi_config = GaudiConfig()
    gaudi_config.use_fused_adam = True
    gaudi_config.use_fused_clip_norm = True

    # Initialize our Trainer
    trainer = GaudiTrainer(
        model=peft_model,
        gaudi_config=gaudi_config,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)

    logs = pd.DataFrame(trainer.state.log_history)
    train_loss = logs[~pd.isna(logs["loss"])]["loss"]
    train_epochs = logs[~pd.isna(logs["loss"])]["epoch"]

    if is_main_process(training_args.local_rank):
        if not os.path.exists("./train_loss"):
            os.makedirs("./train_loss")
        plt.figure(figsize=(10, 5))
        plt.plot(train_epochs, train_loss, label="train loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.savefig(f"./train_loss/train_loss_{training_args.output_dir.split('/')[-1]}.png")
        plt.close()


if __name__ == "__main__":
    main()
