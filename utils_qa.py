# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Post-processing utilities for question answering.
"""
import collections
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import random
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
from transformers import (
    get_linear_schedule_with_warmup,
    AdamW,
    Trainer,
    TrainingArguments,
)
from utils import Statistics, load_json, save_json


logger = logging.getLogger(__name__)

APS = 'answer-position-sentence'
QCSS = 'question-context-similar-sent'
AC = 'answer-candidates'


def get_loss(model, dataloader, task_type):
    loss = 0
    n = 0
    model.eval()
    device = model.device
    for batch in dataloader:
        if task_type == "ex-qa":
            batch = tuple(t.to(device) for t in batch)
            batch_size = batch[0].size(0)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }
        elif task_type == "mc-qa":
            batch_size = batch["input_ids"].size(0)
            for k in batch:
                batch[k] = batch[k].to(device)
            inputs = batch
        else:
            raise ValueError(task_type)
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
        loss_j = outputs.loss.item()
        loss += loss_j * batch_size
        n += batch_size
    loss = loss / n
    return loss


class MyTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            if self.args.warmup_ratio > 0:
                self.args.warmup_steps = int(num_training_steps * self.args.warmup_ratio)
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )
            logger.info(f"num_warmup_steps is set to : {self.args.warmup_steps}")
            logger.info(f"num_training_steps is: {num_training_steps}")

    """
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
        )

        self.log(output.metrics)

        example_ids = [feat.example_id for feat in eval_dataset.features]
        output_pred_file = os.path.join(self.args.output_dir, f"predictions_predict_{self.state.global_step}.json")
        prediction_ids = np.argmax(output.predictions, axis=1).tolist()
        id2pred = {_id: str(pred) for _id, pred in zip(example_ids, prediction_ids)}
        save_json(id2pred, output_pred_file)
        logger.info(f"Saved predictions in {output_pred_file}")

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        return output.metrics
    """


@dataclass
class MyTrainingArguments(TrainingArguments):
    do_biased_train: bool = field(default=False, metadata={"help": ""})
    bias_1: Optional[str] = field(default=None, metadata={"help": ""})
    bias_1_larger_than: Optional[float] = field(default=None, metadata={"help": ""})
    bias_1_smaller_than: Optional[float] = field(default=None, metadata={"help": ""})
    bias_1_included_in: Optional[List[str]] = field(default=None, metadata={"help": ""})
    bias_1_custom_func: Optional[str] = field(default=None, metadata={"help": ""})
    bias_1_same_as: Optional[str] = field(default=None, metadata={"help": ""})
    bias_1_not_equal: Optional[str] = field(default=None, metadata={"help": ""})
    bias_1_top_k: int = field(default=None, metadata={"help": ""})
    bias_2: Optional[str] = field(default=None, metadata={"help": ""})
    bias_2_larger_than: Optional[float] = field(default=None, metadata={"help": ""})
    bias_2_smaller_than: Optional[float] = field(default=None, metadata={"help": ""})
    bias_2_included_in: Optional[List[str]] = field(default=None, metadata={"help": ""})
    bias_2_custom_func: Optional[str] = field(default=None, metadata={"help": ""})
    bias_2_same_as: Optional[str] = field(default=None, metadata={"help": ""})
    bias_2_not_equal: Optional[str] = field(default=None, metadata={"help": ""})
    bias_2_top_k: int = field(default=None, metadata={"help": ""})
    bias_3: Optional[str] = field(default=None, metadata={"help": ""})
    bias_3_larger_than: Optional[float] = field(default=None, metadata={"help": ""})
    bias_3_smaller_than: Optional[float] = field(default=None, metadata={"help": ""})
    bias_3_included_in: Optional[List[str]] = field(default=None, metadata={"help": ""})
    bias_3_custom_func: Optional[str] = field(default=None, metadata={"help": ""})
    bias_3_same_as: Optional[str] = field(default=None, metadata={"help": ""})
    bias_3_not_equal: Optional[str] = field(default=None, metadata={"help": ""})
    bias_3_top_k: int = field(default=None, metadata={"help": ""})
    eval_all_checkpoints: bool = field(default=False, metadata={"help": ""})
    predict_all_checkpoints: bool = field(default=False, metadata={"help": ""})
    do_fewshot_train: bool = field(default=False, metadata={"help": ""})
    num_fewshot_examples: int = field(default=0, metadata={"help": ""})
    do_blend_anti_biased: bool = field(default=False, metadata={"help": ""})
    num_total_examples: int = field(default=None, metadata={"help": ""})
    anti_biased_ratio: float = field(default=0.0, metadata={"help": ""})
    anti_bias_1: Optional[str] = field(default=None, metadata={"help": ""})
    anti_bias_1_larger_than: Optional[float] = field(default=None, metadata={"help": ""})
    anti_bias_1_smaller_than: Optional[float] = field(default=None, metadata={"help": ""})
    anti_bias_1_included_in: Optional[List[str]] = field(default=None, metadata={"help": ""})
    anti_bias_1_custom_func: Optional[str] = field(default=None, metadata={"help": ""})
    anti_bias_1_same_as: Optional[str] = field(default=None, metadata={"help": ""})
    anti_bias_1_not_equal: Optional[str] = field(default=None, metadata={"help": ""})
    anti_bias_1_top_k: int = field(default=None, metadata={"help": ""})
    anti_bias_2: Optional[str] = field(default=None, metadata={"help": ""})
    anti_bias_2_larger_than: Optional[float] = field(default=None, metadata={"help": ""})
    anti_bias_2_smaller_than: Optional[float] = field(default=None, metadata={"help": ""})
    anti_bias_2_included_in: Optional[List[str]] = field(default=None, metadata={"help": ""})
    anti_bias_2_custom_func: Optional[str] = field(default=None, metadata={"help": ""})
    anti_bias_2_same_as: Optional[str] = field(default=None, metadata={"help": ""})
    anti_bias_2_not_equal: Optional[str] = field(default=None, metadata={"help": ""})
    anti_bias_2_top_k: int = field(default=None, metadata={"help": ""})
    anti_bias_3: Optional[str] = field(default=None, metadata={"help": ""})
    anti_bias_3_larger_than: Optional[float] = field(default=None, metadata={"help": ""})
    anti_bias_3_smaller_than: Optional[float] = field(default=None, metadata={"help": ""})
    anti_bias_3_included_in: Optional[List[str]] = field(default=None, metadata={"help": ""})
    anti_bias_3_custom_func: Optional[str] = field(default=None, metadata={"help": ""})
    anti_bias_3_same_as: Optional[str] = field(default=None, metadata={"help": ""})
    anti_bias_3_not_equal: Optional[str] = field(default=None, metadata={"help": ""})
    anti_bias_3_top_k: int = field(default=None, metadata={"help": ""})
    do_online_code: bool = field(default=False, metadata={"help": ""})
    log_before_train: bool = field(default=False, metadata={"help": ""})


def get_bool_of_biased_dataset(data_name, bias, larger_than, smaller_than, included_in, custom_func, same_as, not_equal, top_k, ids, bool_indices, task_type='ex-qa'):
    stat_file = Path(os.environ['RE_ANALYSIS_DIR']) / f"{task_type}/{bias}_{data_name}.json"
    stat = load_json(stat_file)
    assert ((larger_than != None) + (smaller_than != None) + (included_in != None) +\
        (custom_func != None) + (same_as != None)) + (not_equal != None) + (top_k != None) >= 1
    logger.info(" preparing biased training data...")
    if larger_than is not None:
        # We have to use train_features here, rather than train_examples!!
        stat_list = [stat.get(_id, -10000) for _id in ids]
        stat_list = [-10000 if s == None else s for s in stat_list]
        bool_indices *= (torch.tensor(stat_list) >= larger_than)
        logger.info(f" Current size of training set: {bool_indices.sum().item()}")
    if smaller_than is not None:
        stat_list = [stat.get(_id, 10000) for _id in ids]
        stat_list = [10000 if s == None else s for s in stat_list]
        bool_indices *= (torch.tensor(stat_list) <= smaller_than)
        logger.info(f" Current size of training set: {bool_indices.sum().item()}")
    if included_in is not None:
        stat_list = [stat.get(_id, "") for _id in ids]
        stat_list = ["" if s == None else s for s in stat_list]
        bool_indices *= torch.tensor([str(s) in included_in for s in stat_list])
        logger.info(f" Current size of training set: {bool_indices.sum().item()}")
    if custom_func is not None:
        stat_list = [stat.get(_id, []) for _id in ids]
        func = eval(custom_func)
        bool_indices *= torch.tensor([func(s) for s in stat_list])
        logger.info(f" Current size of training set: {bool_indices.sum().item()}")
    if same_as is not None:
        stat_file2 = Path(os.environ['RE_ANALYSIS_DIR']) / f"{task_type}/{same_as}_{data_name}.json"
        stat2 = load_json(stat_file2)
        stat_list = [stat.get(_id, -10000) for _id in ids]
        stat_list2 = [stat2.get(_id, 10000) for _id in ids]
        bool_indices *= torch.tensor([ (s == s2 and s is not None) for s, s2 in zip(stat_list, stat_list2)])
        logger.info(f" Current size of training set: {bool_indices.sum().item()}")
    if not_equal is not None:
        stat_file2 = Path(os.environ['RE_ANALYSIS_DIR']) / f"{task_type}/{not_equal}_{data_name}.json"
        stat2 = load_json(stat_file2)
        stat_list = [stat.get(_id, -10000) for _id in ids]
        stat_list2 = [stat2.get(_id, 10000) for _id in ids]
        bool_indices *= torch.tensor([ (s != s2) and s is not None for s, s2 in zip(stat_list, stat_list2)])
        logger.info(f" Current size of training set: {bool_indices.sum().item()}")
    if top_k is not None:
        stat_items = [(_id, stat.get(_id, -10000)) for _id in list(set(ids))]
        stat_items = [(_id, -10000) if s == None else (_id, s) for _id, s in stat_items]
        stat_items = sorted(stat_items, key=lambda x: x[1])
        selected_ids = [item[0] for item in stat_items[-top_k:]]
        bool_indices *= torch.tensor([_id in selected_ids for _id in ids])
        logger.info(f" Current size of training set: {bool_indices.sum().item()}")
    return bool_indices


def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).

            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
            ``logging`` log level (e.g., ``logging.WARNING``)
    """
    if len(predictions) != 2:
        raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative and min_null_prediction is not None:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if (
            version_2_with_negative
            and min_null_prediction is not None
            and not any(p["offsets"] == (0, 0) for p in predictions)
        ):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def postprocess_qa_predictions_with_beam_search(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    start_n_top: int = 5,
    end_n_top: int = 5,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
):
    """
    Post-processes the predictions of a question-answering model with beam search to convert them to answers that are substrings of the
    original contexts. This is the postprocessing functions for models that return start and end logits, indices, as well as
    cls token predictions.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        start_n_top (:obj:`int`, `optional`, defaults to 5):
            The number of top start logits too keep when searching for the :obj:`n_best_size` predictions.
        end_n_top (:obj:`int`, `optional`, defaults to 5):
            The number of top end logits too keep when searching for the :obj:`n_best_size` predictions.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
            ``logging`` log level (e.g., ``logging.WARNING``)
    """
    if len(predictions) != 5:
        raise ValueError("`predictions` should be a tuple with five elements.")
    start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict() if version_2_with_negative else None

    # Logging.
    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_log_prob = start_top_log_probs[feature_index]
            start_indexes = start_top_index[feature_index]
            end_log_prob = end_top_log_probs[feature_index]
            end_indexes = end_top_index[feature_index]
            feature_null_score = cls_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_start_top`/`n_end_top` greater start and end logits.
            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_index = int(start_indexes[i])
                    j_index = i * end_n_top + j
                    end_index = int(end_indexes[j_index])
                    # Don't consider out-of-scope answers (last part of the test should be unnecessary because of the
                    # p_mask but let's not take any risk)
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue

                    # Don't consider answers with a length negative or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_log_prob[i] + end_log_prob[j_index],
                            "start_log_prob": start_log_prob[i],
                            "end_log_prob": end_log_prob[j_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0:
            # Without predictions min_null_score is going to be None and None will cause an exception later
            min_null_score = -2e-6
            predictions.insert(0, {"text": "", "start_logit": -1e-6, "end_logit": -1e-6, "score": min_null_score})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction and set the probability for the null answer.
        all_predictions[example["id"]] = predictions[0]["text"]
        if version_2_with_negative:
            scores_diff_json[example["id"]] = float(min_null_score)

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions, scores_diff_json
