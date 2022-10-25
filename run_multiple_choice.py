# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""


import logging
import os
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from pathlib import Path
import pickle
import random
import copy

import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.data import Subset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    __version__,
    AdamW,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    get_linear_schedule_with_warmup,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.data.data_collator import default_data_collator
import wandb

from utils_multiple_choice import MultipleChoiceDataset, Split, processors, MULTIPLE_CHOICE_TASKS_NUM_LABELS
from utils_qa import get_bool_of_biased_dataset, MyTrainer, MyTrainingArguments, get_loss
from utils import load_json, save_json

logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def prepare_optimizer_and_schedule(args, model, t_total):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon)
    if args.warmup_ratio > 0:
        args.warmup_steps = int(t_total * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    return optimizer, scheduler


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    if __version__ < '4.0.0':
        warmup_ratio: float = field(default=0.0, metadata={"help": "ratio of warmup steps to total steps"})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info(f"Huggingface transformers version: {__version__}")
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False
    )

    # Get datasets
    train_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train or training_args.do_online_code
        else None
    )
    eval_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval or training_args.eval_all_checkpoints
        else None
    )
    predict_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )
        if training_args.do_predict or training_args.predict_all_checkpoints
        else None
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": simple_accuracy(preds, p.label_ids)}

    train_batch_size = training_args.per_device_train_batch_size * max(1, training_args.n_gpu)

    if __version__ < '4.0.0':
        if data_args.warmup_ratio > 0:
            training_args.warmup_ratio = data_args.warmup_ratio
        else:
            training_args.warmup_ratio = -1

    if training_args.do_biased_train or training_args.do_fewshot_train:
        bool_indices = torch.ones(len(train_dataset))

    if training_args.do_biased_train:
        train_ids = [feat.example_id for feat in train_dataset.features]
        logger.info(f" Current size of training set: {bool_indices.sum().item()}")
        data_name = f'{data_args.task_name}-train'

        if training_args.bias_1 is not None:
            bool_indices = get_bool_of_biased_dataset(
                data_name,
                training_args.bias_1,
                training_args.bias_1_larger_than,
                training_args.bias_1_smaller_than,
                training_args.bias_1_included_in,
                training_args.bias_1_custom_func,
                training_args.bias_1_same_as,
                training_args.bias_1_not_equal,
                training_args.bias_1_top_k,
                train_ids,
                bool_indices,
                task_type='mc-qa')
        if training_args.bias_2 is not None:
            bool_indices = get_bool_of_biased_dataset(
                data_name,
                training_args.bias_2,
                training_args.bias_2_larger_than,
                training_args.bias_2_smaller_than,
                training_args.bias_2_included_in,
                training_args.bias_2_custom_func,
                training_args.bias_2_same_as,
                training_args.bias_2_not_equal,
                training_args.bias_2_top_k,
                train_ids,
                bool_indices,
                task_type='mc-qa')
        if training_args.bias_3 is not None:
            bool_indices = get_bool_of_biased_dataset(
                data_name,
                training_args.bias_3,
                training_args.bias_3_larger_than,
                training_args.bias_3_smaller_than,
                training_args.bias_3_included_in,
                training_args.bias_3_custom_func,
                training_args.bias_3_same_as,
                training_args.bias_3_not_equal,
                training_args.bias_3_top_k,
                train_ids,
                bool_indices,
                task_type='mc-qa')

    if training_args.do_fewshot_train:
        train_qas_id = list(set([feat.example_id for feat, b in zip(train_dataset.features, bool_indices) if b.item() == 1]))
        n_total_train = len(train_qas_id)
        logger.info(f"The total number of unique examples: {n_total_train}")
        fewshot_qas_id = random.sample(train_qas_id, training_args.num_fewshot_examples)
        bool_indices *= torch.tensor([feat.example_id in fewshot_qas_id for feat in train_dataset.features])

    if training_args.do_blend_anti_biased and training_args.do_biased_train:
        logger.info(f"The total number of training examples should be: {training_args.num_total_examples}")
        num_anti_biased = math.ceil(training_args.anti_biased_ratio * training_args.num_total_examples)
        num_biased = training_args.num_total_examples - num_anti_biased
        logger.info(f"The number of biased examples: {num_biased}")
        logger.info(f"The number of anti-biased examples: {num_anti_biased}")

        # fewshot unique dataset
        train_qas_id = list(set([feat.example_id for feat, b in zip(train_dataset.features, bool_indices) if b.item() == 1]))
        n_total_train = len(train_qas_id)
        logger.info(f"The total number of unique examples: {n_total_train}")
        biased_qas_id = random.sample(train_qas_id, num_biased)
        bool_indices *= torch.tensor([feat.example_id in biased_qas_id for feat in train_dataset.features])
        logger.info(f" The size of the biased training set: {bool_indices.sum().item()}")

        anti_bool_indices = torch.ones(len(train_dataset))
        train_ids = [feat.example_id for feat in train_dataset.features]
        logger.info(f" Current size of anti biased training set: {anti_bool_indices.sum().item()}")

        if training_args.anti_bias_1 is not None:
            anti_bool_indices = get_bool_of_biased_dataset(
                data_name,
                training_args.anti_bias_1,
                training_args.anti_bias_1_larger_than,
                training_args.anti_bias_1_smaller_than,
                training_args.anti_bias_1_included_in,
                training_args.anti_bias_1_custom_func,
                training_args.anti_bias_1_same_as,
                training_args.anti_bias_1_not_equal,
                training_args.anti_bias_1_top_k,
                train_ids,
                anti_bool_indices,
                task_type='mc-qa')
        if training_args.anti_bias_2 is not None:
            anti_bool_indices = get_bool_of_biased_dataset(
                data_name,
                training_args.anti_bias_2,
                training_args.anti_bias_2_larger_than,
                training_args.anti_bias_2_smaller_than,
                training_args.anti_bias_2_included_in,
                training_args.anti_bias_2_custom_func,
                training_args.anti_bias_2_same_as,
                training_args.anti_bias_2_not_equal,
                training_args.anti_bias_2_top_k,
                train_ids,
                anti_bool_indices,
                task_type='mc-qa')
        if training_args.anti_bias_3 is not None:
            anti_bool_indices = get_bool_of_biased_dataset(
                data_name,
                training_args.anti_bias_3,
                training_args.anti_bias_3_larger_than,
                training_args.anti_bias_3_smaller_than,
                training_args.anti_bias_3_included_in,
                training_args.anti_bias_3_custom_func,
                training_args.anti_bias_3_same_as,
                training_args.anti_bias_3_not_equal,
                training_args.anti_bias_3_top_k,
                train_ids,
                anti_bool_indices,
                task_type='mc-qa')

        train_qas_id = list(set([feat.example_id for feat, b in zip(train_dataset.features, anti_bool_indices) if b.item() == 1]))
        n_total_train = len(train_qas_id)
        logger.info(f"The total number of unique anti biased examples: {n_total_train}")
        anti_biased_qas_id = random.sample(train_qas_id, num_anti_biased)
        anti_bool_indices *= torch.tensor([feat.example_id in anti_biased_qas_id for feat in train_dataset.features])
        logger.info(f" The size of the anti biased training set: {anti_bool_indices.sum().item()}")
        bool_indices += anti_bool_indices

    if training_args.do_biased_train or training_args.do_fewshot_train:
        indices = torch.nonzero(bool_indices, as_tuple=False).view(-1)
        train_dataset = Subset(train_dataset, indices)
        logger.info(f" The size of the resulting training set: {len(train_dataset)}")

    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        name=Path(training_args.output_dir).name,
        config={
            "epochs": training_args.num_train_epochs,
            "train_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
            "warmup_ratio": training_args.warmup_ratio,
            "model_name_or_path": model_args.model_name_or_path,
        }
    )

    # Training
    if training_args.do_train:
        model = AutoModelForMultipleChoice.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

        # Initialize our Trainer
        trainer = MyTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=predict_dataset,
            compute_metrics=compute_metrics,
        )

        if training_args.log_before_train:
            evaluate(trainer, predict_dataset, training_args, wandb=wandb, desc="predict", step="0")

        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_online_code:
        Path(training_args.output_dir).mkdir(exist_ok=True)
        # Added here for reproductibility
        set_seed(training_args.seed)
        shuffled_dataset = Subset(
            train_dataset, indices=np.random.permutation(len(train_dataset)))
        fractions = [0.001,0.002,0.004,0.008,0.016,0.032,0.0625,0.125,0.25,0.5,1]
        val_fraction = 0.1
        total_len = len(shuffled_dataset)
        logger.info(f"Total number of examples for online code: {total_len}")
        max_num_epochs = 10
        max_train_steps = 500
        patience = -1
        max_loss = - math.log(1 / 4) # loss of uniform distribution
        eval_loss = max_loss
        nlls = []
        nlls.append(eval_loss)
        train_batch_size = training_args.per_device_train_batch_size * max(1, training_args.n_gpu)
        eval_batch_size = training_args.per_device_eval_batch_size * max(1, training_args.n_gpu)

        train_portions = []
        eval_portions = []
        while math.ceil(fractions[0] * total_len) <= 1:
            fractions = fractions[1:]
        block_sizes = [math.ceil(fractions[0] * total_len)]
        for i in range(len(fractions) - 1):
            block_size = math.ceil(fractions[i + 1] * total_len) - math.ceil(fractions[i] * total_len)
            train_portions.append(Subset(shuffled_dataset,
                                  range(0, math.ceil(fractions[i] * total_len))))
            eval_portions.append(Subset(shuffled_dataset,
                                 range(math.ceil(fractions[i] * total_len), math.ceil(fractions[i + 1] * total_len))))
            block_sizes.append(block_size)
        logger.info(f"Sizes of the training sets in the blocks are: {', '.join([str(len(tr)) for tr in train_portions])}")
        for i in range(len(train_portions)):
            dev_dataset_i = Subset(train_portions[i], range(0, math.ceil(len(train_portions[i]) * val_fraction)))
            dev_sampler = SequentialSampler(dev_dataset_i)
            dev_dataloader = DataLoader(
                dev_dataset_i,
                sampler=dev_sampler,
                batch_size=eval_batch_size,
                collate_fn=default_data_collator)
            train_dataset_i = Subset(train_portions[i], range(math.ceil(len(train_portions[i]) * val_fraction), len(train_portions[i])))
            train_sampler = RandomSampler(train_dataset_i)
            train_dataloader = DataLoader(
                train_dataset_i,
                sampler=train_sampler,
                batch_size=train_batch_size,
                collate_fn=default_data_collator)
            eval_sampler = SequentialSampler(eval_portions[i])
            eval_dataloader = DataLoader(
                eval_portions[i],
                sampler=eval_sampler,
                batch_size=eval_batch_size,
                collate_fn=default_data_collator)

            t_total = len(train_dataloader) // training_args.gradient_accumulation_steps * max_num_epochs
            t_total = max(t_total, max_train_steps)
            num_epochs = t_total // (len(train_dataloader) // training_args.gradient_accumulation_steps) + 1
            logging_online_steps = 20
            min_dev_loss = dev_loss = max_loss
            global_step = 0
            early_stop = False
            num_worse_step = 0

            # re-initializaion
            model = AutoModelForMultipleChoice.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )
            model.to(training_args.device)
            model.zero_grad()
            train_iterator = trange(
                0, int(num_epochs), desc="Epoch"
            )

            # Prepare optimizer and schedule (linear warmup and decay)
            optimizer, scheduler = prepare_optimizer_and_schedule(training_args, model, t_total)
            best_checkpoint = copy.deepcopy(model.state_dict())

            for _ in train_iterator:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration")
                for step, batch in enumerate(epoch_iterator):
                    model.train()
                    for k in batch:
                        batch[k] = batch[k].to(training_args.device)
                    outputs = model(**batch, return_dict=True)
                    # model outputs are always tuple in transformers (see doc)
                    loss = outputs.loss
                    if training_args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                    if training_args.gradient_accumulation_steps > 1:
                        loss = loss / training_args.gradient_accumulation_steps
                    if training_args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    if (step + 1) % training_args.gradient_accumulation_steps == 0:
                        if training_args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), training_args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)

                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad()
                        global_step += 1
                        if training_args.local_rank in [-1, 0]:
                            metrics = {}
                            metrics["online_code/lr"] = scheduler.get_lr()[0]
                            metrics["online_code/train_loss"] = loss
                            if global_step % logging_online_steps == 0:
                                dev_loss = get_loss(model, dev_dataloader, "mc-qa")
                                if dev_loss < min_dev_loss - 0.001:
                                    min_dev_loss = dev_loss
                                    best_checkpoint = copy.deepcopy(model.state_dict())
                                else:
                                    # stop_block
                                    num_worse_step += 1

                                if num_worse_step >= patience and patience > 0:
                                    if best_checkpoint is not None:
                                        model.load_state_dict(best_checkpoint)
                                    else:
                                        logger.warning("best_checkpoint is None")
                                    eval_loss = get_loss(model, eval_dataloader, "mc-qa")
                                    nlls.append(eval_loss)
                                    early_stop = True
                            metrics["online_code/dev_loss"] = dev_loss
                            metrics["online_code/eval_loss"] = eval_loss
                            wandb.log(metrics)
                    if global_step >= t_total or early_stop:
                        epoch_iterator.close()
                        break
                if global_step >= t_total or early_stop:
                    train_iterator.close()
                    break
            if not early_stop:
                if best_checkpoint is not None:
                    model.load_state_dict(best_checkpoint)
                else:
                    logger.warning("best_checkpoint is None")
                eval_loss = get_loss(model, eval_dataloader, "mc-qa")
                nlls.append(eval_loss)
            del model
        codelengths = [nll / math.log(2) for nll in nlls]
        if len(codelengths) != len(block_sizes):
            logger.warning("len(codelengths) != len(block_sizes)")
        mdl = sum([block_size * per_sample_codelength for block_size, per_sample_codelength in zip(block_sizes, codelengths)])
        wandb.log({"online_code/mdl": mdl})
        output_online_code = {
            "MDL": mdl,
            "nll": nlls,
            "block_size": block_sizes,
            "codelength": codelengths,
        }
        with open(Path(training_args.output_dir) / "online_code.pkl", "wb") as f:
            pickle.dump(output_online_code, f)
        save_json(output_online_code, Path(training_args.output_dir) / "online_code.json")
        logger.info(f"Saved result files {str(Path(training_args.output_dir) / 'online_code.json')}")


    # Evaluation
    if training_args.do_eval or training_args.do_predict:
        model = AutoModelForMultipleChoice.from_pretrained(
            training_args.output_dir)

        # Initialize our Trainer
        trainer = MyTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

    if training_args.do_eval:
        evaluate(trainer, eval_dataset,  training_args, wandb=wandb, desc="eval")

    # Prediction
    if training_args.do_predict:
        evaluate(trainer, predict_dataset, training_args, wandb=wandb, desc="predict")

    if training_args.eval_all_checkpoints or training_args.predict_all_checkpoints:
        checkpoints = sorted(Path(training_args.output_dir).glob("checkpoint-*"))
        for checkpoint in checkpoints:
            step = checkpoint.name.split("-")[1]
            model = AutoModelForMultipleChoice.from_pretrained(checkpoint)
            trainer = MyTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
            )
            if training_args.eval_all_checkpoints:
                evaluate(trainer, eval_dataset, training_args, wandb=wandb, desc="eval", step=step)
            if training_args.predict_all_checkpoints:
                evaluate(trainer, predict_dataset, training_args, wandb=wandb, desc="predict", step=step)

def evaluate(trainer, dataset, training_args, wandb=None, desc="", step=""):
    logger.info("*** Evaluate ***")

    example_ids = [feat.example_id for feat in dataset.features]
    output = trainer.predict(dataset)

    output_eval_file = os.path.join(training_args.output_dir, f"{desc}_results_{step}.txt")
    result = output.metrics
    log = {}
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key, value in result.items():
            logger.info("  %s = %s", key, value)
            writer.write("%s = %s\n" % (key, value))
            if step == "":
                log[f"{desc}/{key}"] = value
            else:
                log[f"{desc}/{key}_dynamics"] = value
    if wandb is not None:
        if step == "":
            wandb.log(log)
        else:
            wandb.log(log, step=int(step))

    output_pred_file = os.path.join(training_args.output_dir, f"predictions_{desc}_{step}.json")
    prediction_ids = np.argmax(output.predictions, axis=1).tolist()
    id2pred = {_id: str(pred) for _id, pred in zip(example_ids, prediction_ids)}
    save_json(id2pred, output_pred_file)
    logger.info(f"Saved predictions in {output_pred_file}")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
