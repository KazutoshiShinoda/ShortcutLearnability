import numpy as np
from collections import Counter, OrderedDict
import string
import re
import argparse
import json
import sys
import math


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def squad_exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions, stratify=False, return_dict=False):
    if stratify:
        f1 = {}
        exact_match = {}
        q_types = {} # count
    else:
        f1 = exact_match = 0
    if return_dict:
        score_dict = {}
    not_exist_in_pred = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if qa['id'] not in predictions:
                    #message = 'Unanswered question ' + qa['id'] + \
                    #          ' will not receive score 0 to evaluate with split'
                    #print(message, file=sys.stderr)
                    not_exist_in_pred += 1
                    continue
                total += 1
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                if stratify:
                    q_type = qa['question_tokenized'][0].lower()
                    q_types[q_type] = q_types.get(q_type, 0) + 1
                _exact_match = metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                _f1 = metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)
                if stratify:
                    exact_match[q_type] = exact_match.get(q_type, 0) + _exact_match
                    f1[q_type] = f1.get(q_type, 0) + _f1
                else:
                    exact_match += _exact_match
                    f1 += _f1
                if return_dict:
                    score_dict[qa['id']] = {
                        'F1': _f1,
                        'EM': _exact_match
                    }
    if stratify:
        for q_type in f1:
            f1[q_type] = 100.0 * f1[q_type] / q_types[q_type]
            exact_match[q_type] = 100.0 * exact_match[q_type] / q_types[q_type]
    else:
        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total
    # print('# of unanswerd questions: {}'.format(not_exist_in_pred))
    # print('# of answered questions: {}'.format(total))
    if stratify:
        all_score = {'exact_match': exact_match, 'f1': f1, 'q_type': q_types}
    else:
        all_score = {'exact_match': exact_match, 'f1': f1}
    if return_dict:
        return all_score, score_dict
    return all_score
