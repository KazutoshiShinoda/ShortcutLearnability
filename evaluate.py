from __future__ import print_function
import numpy as np
from collections import Counter, OrderedDict
import string
import re
import argparse
import json
import sys
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk import ngrams
from nlgeval import NLGEval
import math

ORIGINAL_DEV_FILE = '/data/shinoda/dataset/qa/squad-du-split/dev.json'
EASY_FILE = '/home/shinoda/git/QuestionGeneration/input/mrc-heuristics/subsets/squad-easy-subset.json'
HARD_FILE = '/home/shinoda/git/QuestionGeneration/input/mrc-heuristics/subsets/squad-hard-subset.json'

def load_json(file):
    with open(file, 'r') as f:
        d = json.load(f)
    return d


def evaluate_easy_hard(prediction):
    dev = load_json(ORIGINAL_DEV_FILE)
    _, score_dict = evaluate(dev['data'], prediction, return_dict=True)
    easy_subset = load_json(EASY_FILE)
    hard_subset = load_json(HARD_FILE)

    easy_f1 = 0
    easy_em = 0
    hard_f1 = 0
    hard_em = 0
    n_easy = 0
    n_hard = 0
    n_error = 0
    for key in score_dict:
        if key in easy_subset:
            n_easy += 1
            easy_f1 += score_dict[key]['F1']
            easy_em += score_dict[key]['EM']
        elif key in hard_subset:
            n_hard += 1
            hard_f1 += score_dict[key]['F1']
            hard_em += score_dict[key]['EM']
        else:
            n_error += 1

    if n_easy != len(easy_subset):
        print("Missing some easy examples:", n_easy, len(easy_subset))
    if n_hard != len(hard_subset):
        print("Missing some hard examples:", n_hard, len(hard_subset))
    if n_error > 0:
        print(f"{n_error} examples were not found in any subsets.")
    easy_f1 = round(easy_f1 * 100 / n_easy, 2)
    easy_em = round(easy_em * 100 / n_easy, 2)
    hard_f1 = round(hard_f1 * 100 / n_hard, 2)
    hard_em = round(hard_em * 100 / n_hard, 2)
    return f"{easy_em:.2f}/{easy_f1:.2f}\t{hard_em:.2f}/{hard_f1:.2f}"


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


def squad_f1_score(prediction, ground_truth):
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


def squad_precision_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    return precision


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


def evaluate_adversarial(dataset, predictions, verbose=False, id_set=None):
    orig_f1_score = 0.0
    orig_exact_match_score = 0.0
    adv_f1_scores = {}  # Map from original ID to F1 score
    adv_exact_match_scores = {}  # Map from original ID to exact match score
    adv_ids = {}
    all_ids = set()  # Set of all original IDs
    f1 = exact_match = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                orig_id = qa['id'].split('-')[0]
                if id_set and orig_id not in id_set: continue
                all_ids.add(orig_id)
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + ' will receive score 0.'
                    print(message)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                cur_exact_match = metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                cur_f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
                if orig_id == qa['id']:
                    # This is an original example
                    orig_f1_score += cur_f1
                    orig_exact_match_score += cur_exact_match
                    if orig_id not in adv_f1_scores:
                        # Haven't seen adversarial example yet, so use original for adversary
                        adv_ids[orig_id] = orig_id
                        adv_f1_scores[orig_id] = cur_f1
                        adv_exact_match_scores[orig_id] = cur_exact_match
                else:
                    # This is an adversarial example
                    if (orig_id not in adv_f1_scores or adv_ids[orig_id] == orig_id
                            or adv_f1_scores[orig_id] > cur_f1):
                        # Always override if currently adversary currently using orig_id
                        adv_ids[orig_id] = qa['id']
                        adv_f1_scores[orig_id] = cur_f1
                        adv_exact_match_scores[orig_id] = cur_exact_match
    if verbose:
        print_details(dataset, predictions, adv_ids)
    orig_f1 = 100.0 * orig_f1_score / len(all_ids)
    orig_exact_match = 100.0 * orig_exact_match_score / len(all_ids)
    adv_exact_match = 100.0 * sum(adv_exact_match_scores.values()) / len(all_ids)
    adv_f1 = 100.0 * sum(adv_f1_scores.values()) / len(all_ids)
    return OrderedDict([
        ('orig_exact_match', orig_exact_match),
        ('orig_f1', orig_f1),
        ('adv_exact_match', adv_exact_match),
        ('adv_f1', adv_f1),
    ])


"""
def f1_score(pred_span, true_span):
    Not precise

    Parameters
    ----------
    pred_span : list of int
        like [p_s, p_e]
    true_span : list of int
        like [t_s, t_e]

    p_s = pred_span[0]
    p_e = pred_span[1]
    t_s = true_span[0]
    t_e = true_span[1]

    if t_e < p_s or p_e < t_s or p_e < p_s or t_e < t_s:
        return 0
    elif t_e <= p_e:
        if t_s <= p_s:
            num_same = t_e - p_s + 1
        else:
            num_same = t_e - t_s + 1
    elif p_e <= t_e:
        if p_s <= t_s:
            num_same = p_e - t_s + 1
        else:
            num_same = p_e - p_s + 1
    precision = 1.0 * num_same / (p_e - p_s + 1)
    recall = 1.0 * num_same / (t_e - t_s + 1)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
"""
"""
def exact_match_score(pred_span, true_span):
    Not precise

    Parameters
    ----------
    pred_span : list of int
        like [p_s, p_e]
    true_span : list of int
        like [t_s, t_e]

    p_s = pred_span[0]
    p_e = pred_span[1]
    t_s = true_span[0]
    t_e = true_span[1]

    return (p_s == t_s) and (p_e == t_e)
"""

def multi_f1_em_score(pred, true):
    """
    pred : list of str [pred1, pred2, ...]
    true : list of str [true1, true2, ...]
    """
    n_t = len(true)
    n_p = len(pred)
    f1_matrix = np.zeros((n_t, n_p))
    em_matrix = np.zeros((n_t, n_p))
    for i in range(n_t):
        t = true[i]
        for j in range(n_p):
            p = pred[j]
            f1_matrix[i, j] += squad_f1_score(' '.join(p), ' '.join(t))
            em_matrix[i, j] += squad_exact_match_score(' '.join(p), ' '.join(t))
    f1 = np.max(f1_matrix, axis=1)
    em = np.max(em_matrix, axis=1)
    return f1.sum(), em.sum()


def multi_prop_bin_em_score(pred, true):
    """
    pred : list of str [pred1, pred2, ...]
    true : list of str [true1, true2, ...]
    """
    n_t = len(true)
    n_p = len(pred)
    prop_matrix = np.zeros((n_t, n_p))
    bin_matrix = np.zeros((n_t, n_p))
    em_matrix = np.zeros((n_t, n_p))
    for i in range(n_t):
        t = true[i]
        for j in range(n_p):
            p = pred[j]
            prop_matrix[i, j] += squad_precision_score(' '.join(p), ' '.join(t))
            bin_matrix[i, j] += int(prop_matrix[i, j] > 0)
            em_matrix[i, j] += squad_exact_match_score(' '.join(p), ' '.join(t))
    p_prop = np.mean(np.max(prop_matrix, axis=0))
    r_prop = np.mean(np.max(prop_matrix, axis=1))
    if p_prop * r_prop == 0:
        f_prop = 0
    else:
        f_prop = 2 * p_prop * r_prop / (p_prop + r_prop)
    p_bin = np.mean(np.max(bin_matrix, axis=0))
    r_bin = np.mean(np.max(bin_matrix, axis=1))
    if p_bin * r_bin == 0:
        f_bin = 0
    else:
        f_bin = 2 * p_bin * r_bin / (p_bin + r_bin)
    p_em = np.mean(np.max(em_matrix, axis=0))
    r_em = np.mean(np.max(em_matrix, axis=1))
    if p_em * r_em == 0:
        f_em = 0
    else:
        f_em = 2 * p_em * r_em / (p_em + r_em)
    return p_prop, r_prop, f_prop, p_bin, r_bin, f_bin, p_em, r_em, f_em


def bleu_score(pred, true, n=1, smoothing=0):
    """Calculate BLEU-N score

    pred : list of str [w1, w2, ...]
    true : list of str [w1, w2, ...]
    """
    weights = tuple((1. / n for _ in range(n)))
    if smoothing == 0:
        smoothing_function = SmoothingFunction().method0
    elif smoothing == 1:
        smoothing_function = SmoothingFunction().method1
    elif smoothing == 2:
        smoothing_function = SmoothingFunction().method2
    elif smoothing == 3:
        smoothing_function = SmoothingFunction().method3
    elif smoothing == 4:
        smoothing_function = SmoothingFunction().method4
    elif smoothing == 5:
        smoothing_function = SmoothingFunction().method5
    elif smoothing == 6:
        smoothing_function = SmoothingFunction().method6
    elif smoothing == 7:
        smoothing_function = SmoothingFunction().method7
    score = sentence_bleu([true], pred, weights=weights,
                          smoothing_function=smoothing_function)
    return score

def self_bleu_score(preds, n=1):
    weights = tuple((1. / n for _ in range(n)))
    scores = 0
    for i in range(len(preds)):
        ref = preds[:i] + preds[i+1:]
        hyp = preds[i]
        scores += sentence_bleu(ref, hyp, weights=weights,
                                smoothing_function=SmoothingFunction().method1)
    score = scores / len(preds)
    return score


def multi_bleu_score(pred, true, n=[1, 2]):
    """
    pred : (n_answers, n_samples, q_length)
    true : (n_answers, q_length)
    """
    n_t = len(true)
    n_p = len(pred[0])
    bleu_matrices = [np.zeros((n_t, n_p)) for k in n]
    max_score = [0] * len(n)
    mean_score = [0] * len(n)
    for k, n_k in enumerate(n):
        for i in range(n_t):
            for j in range(n_p):
                bleu_matrices[k][i][j] += bleu_score(pred[i][j], true[i], n=n_k)
        max_score[k] = np.sum(np.max(bleu_matrices[k], axis=1))
        mean_score[k] = np.sum(np.mean(bleu_matrices[k], axis=1))
    return max_score, mean_score


def dist_score(seqs, n=[1, 2, 3, 4]):
    """Dist-n

    Defined as the number of distinctive n-grams
    divided by the total number of n-grams

    Params
    ------
    seqs : (n_seqs, seq_length)

    Return
    ------
    outputs : (len(n),)
    """
    outputs = []
    for n_k in n:
        all_gram = []
        for seq in seqs:
            res = ngrams(seq, n_k)
            res = list(res)
            all_gram = all_gram + res
        if len(all_gram) == 0:
            # to avoid zero division error due to sequences of low length
            outputs.append(0)
        else:
            outputs.append(len(set(all_gram)) / len(all_gram))
    return outputs

def ent_and_abs_dist(corpus, n):
    """
    Args:
        corpus: list of {list of words(str) | tokenized sentence}
        n: int, n-gram
    Return:
        ent_bit: float, entropy (bit)
        n_unique: int, number of unique n-grams
    """
    ngram_list = []
    for s in corpus:
        ngram_list += list(ngrams(s, n))
    count = Counter(ngram_list)
    n_unique = len(count)
    count_item = count.items()
    all_count = sum(map(lambda x: x[1], count_item))
    ent_bit = - (1/all_count) * sum(map(lambda x: x[1] * math.log(x[1]/all_count, 2), count_item))
    return ent_bit, n_unique

class QualityEval(object):
    def __init__(self, no_skipthoughts=True, metrics=[
            'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR',
            'ROUGE_L', 'EmbeddingAverageCosineSimilairty']):
        assert len(metrics) != 0
        self.no_skipthoughts = no_skipthoughts
        self.nlgeval = NLGEval(no_skipthoughts=no_skipthoughts)
        self.metrics = metrics
        self.stop_words = ['\n']

    def multi_score(self, pred, true):
        """
        pred (list): [n_answers, n_samples, q_length]
        true (list): [n_answers, q_length]
        max_score (list): len == len(self.metrics)
        """
        n_t = len(true)
        n_p = len(pred[0])
        metric_matrices = [np.zeros((n_t, n_p)) for _ in self.metrics]
        max_score = [0] * len(self.metrics)
        mean_score = [0] * len(self.metrics)
        for i in range(n_t):
            t = self._remove_stop_words(true[i])
            for j in range(n_p):
                p = self._remove_stop_words(pred[i][j])
                out = self.nlgeval.compute_individual_metrics([' '.join(t)], ' '.join(p))
                for k, metric in enumerate(self.metrics):
                    metric_matrices[k][i][j] += out[metric]
        for k in range(len(self.metrics)):
            max_score[k] = np.sum(np.max(metric_matrices[k], axis=1))
            mean_score[k] = np.sum(np.mean(metric_matrices[k], axis=1))
        return max_score, mean_score

    def _remove_stop_words(self, seq):
        """
        seq (list)
        """
        s = seq.copy()
        for sw in self.stop_words:
            s = filter(lambda x: not sw in x, s)
        return list(s)

    def restart(self):
        self.nlgeval = NLGEval(no_skipthoughts=self.no_skipthoughts)
