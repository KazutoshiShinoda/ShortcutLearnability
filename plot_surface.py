from transformers import (
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
    set_seed,
)
from transformers.data.data_collator import default_data_collator
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
    get_raw_scores
)
import json
import time
import random
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from squad import SquadResult, SquadV1Processor, SquadV2Processor, SquadExample, squad_convert_example_to_features
from evaluate import evaluate
from utils_qa import get_bool_of_biased_dataset, MyTrainer, get_loss
from utils_multiple_choice import MultipleChoiceDataset, Split, processors
import seaborn as sns
sns.set()

device =  torch.device("cuda")

X_DIRECTION = 'x_direction.pt'
Y_DIRECTION = 'y_direction.pt'
OUTPUT_PDF_FILE = 'surface.pdf'

dataset_dir = Path('dataset')

def state2vector(state, shapes):
    vector = torch.tensor([])
    for k in shapes:
        vector = torch.cat([vector, state[k].view(-1)], dim=0)
    return vector

def vector2model(vector, shapes, model_i):
    state_dict = model_i.state_dict()

    for k, v in model_i.named_parameters():
        start_index = shapes[k]['start_index']
        shape = shapes[k]['shape']
        w = vector[start_index:start_index + np.prod(shape)]
        w = w.view(*list(shape))
        state_dict[k] = w

    model_i.load_state_dict(state_dict)


def prepare_expa_dataloader(task_name, tokenizer, max_seq_length, doc_stride, max_query_length, batch_size):
    if task_name.lower() == "squad":
        cached_features_file = dataset_dir / "ex-qa/squad/cached_train-v1.1_bert-base-uncased_384"
        data_name = "train-v1.1"
        train_file = dataset_dir / "ex-qa/squad/train-v1.1.json"
    elif task_name.lower() == "nq":
        cached_features_file = dataset_dir / "ex-qa/mrqa/train/cached_NaturalQuestionsShort-train-from-MRQA_bert-base-uncased_384"
        data_name = "NaturalQuestionsShort-train-from-MRQA"
        train_file = dataset_dir / "ex-qa/mrqa/train/NaturalQuestionsShort-train-from-MRQA.json"
    else:
        raise ValueError(task_name)
    if os.path.exists(cached_features_file):
        features_and_dataset = torch.load(cached_features_file)
        train_features, train_dataset, train_examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        train_examples = SquadV1Processor().get_train_examples(
            None, filename=train_file)
        train_features, train_dataset = squad_convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=True,
            return_dataset="pt",
            threads=8
        )

    bool_indices = torch.ones(len(train_features))

    def get_bool_of_biased_dataset(bias, larger_than, smaller_than, included_in, custom_func, bool_indices):
        stat_file = Path(os.environ['RE_ANALYSIS_DIR']) / f"ex-qa/{bias}_{data_name}.json"
        with open(stat_file, 'r') as f:
            stat = json.load(f)
        assert ((larger_than != None) + (smaller_than != None) + (included_in != None) + (custom_func != None)) >= 1
        if larger_than is not None:
            # We have to use train_features here, rather than train_examples!!
            stat_list = [stat.get(feat.qas_id, -10000) for feat in train_features]
            stat_list = [-10000 if s == None else s for s in stat_list]
            bool_indices *= (torch.tensor(stat_list) >= larger_than)
        if smaller_than is not None:
            stat_list = [stat.get(feat.qas_id, 10000) for feat in train_features]
            stat_list = [10000 if s == None else s for s in stat_list]
            bool_indices *= (torch.tensor(stat_list) <= smaller_than)
        if included_in is not None:
            stat_list = [stat.get(feat.qas_id, "") for feat in train_features]
            stat_list = ["" if s == None else s for s in stat_list]
            bool_indices *= torch.tensor([str(s) in included_in for s in stat_list])
        if custom_func is not None:
            stat_list = [stat.get(feat.qas_id, []) for feat in train_features]
            func = eval(custom_func)
            bool_indices *= torch.tensor([func(s) for s in stat_list])
        return bool_indices

    bool_indices = get_bool_of_biased_dataset(
        'answer-position-sentence',
        None,
        None,
        ['0'],
        None,
        bool_indices)
    bool_indices = get_bool_of_biased_dataset(
        'question-context-similar-sent',
        None,
        None,
        ['0'],
        None,
        bool_indices)
    bool_indices = get_bool_of_biased_dataset(
        'answer-candidates',
        None,
        None,
        ['1'],
        None,
        bool_indices)
    indices = torch.nonzero(bool_indices, as_tuple=False).view(-1)
    train_dataset = Subset(train_dataset, indices)

    print(f'Num. of examples: {len(train_dataset)}')

    if task_name.lower() == "nq":
        random.seed(42) # for reproducibility
        idx = random.sample(list(range(len(train_dataset))), 2000)
        train_dataset = Subset(train_dataset, idx)
        print(f" The size of the re-sampled dataset from NQ: {len(train_dataset)}")

    sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size)
    return train_dataloader


def prepare_mcqa_dataloader(task_name, tokenizer, max_seq_length, batch_size):
    if task_name.lower() == "race":
        data_dir = dataset_dir / "mc-qa/RACE"
        data_name = "race-train"
    elif task_name.lower() == "reclor":
        data_dir = dataset_dir / "mc-qa/reclor"
        data_name = "reclor-train"
    train_dataset = MultipleChoiceDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        task=task_name.lower(),
        max_seq_length=max_seq_length,
        overwrite_cache=False,
        mode=Split.train,
    )
    bool_indices = torch.ones(len(train_dataset))
    train_ids = [feat.example_id for feat in train_dataset.features]
    print(f" Current size of training set: {bool_indices.sum().item()}")

    bool_indices = get_bool_of_biased_dataset(
        data_name,
        "correct-has-max-lexical-overlap",
        None,
        None,
        "1",
        None,
        None,
        None,
        train_ids,
        bool_indices,
        task_type='mc-qa')
    bool_indices = get_bool_of_biased_dataset(
        data_name,
        "only-correct-has-top1-word",
        None,
        None,
        "1",
        None,
        None,
        None,
        train_ids,
        bool_indices,
        task_type='mc-qa')
    indices = torch.nonzero(bool_indices, as_tuple=False).view(-1)
    train_dataset = Subset(train_dataset, indices)
    print(f" The size of the resulting dataset: {len(train_dataset)}")
    if task_name.lower() == "race":
        random.seed(42) # for reproducibility
        idx = random.sample(list(range(len(train_dataset))), 500)
        train_dataset = Subset(train_dataset, idx)
        print(f" The size of the re-sampled dataset from RACE: {len(train_dataset)}")
    sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=default_data_collator,
        num_workers=0,
        pin_memory=True,
    )
    return train_dataloader


def prepare_surface(vector0, delta1, delta2, shapes, model_i, dataloader, xmin=-1, ymin=-1, xmax=1, ymax=1, width=101):
    model_class = str(type(model_i))
    if "QuestionAnswering" in model_class:
        task_type = "ex-qa"
    elif "MultipleChoice" in model_class:
        task_type = "mc-qa"
    else:
        raise ValueError(model_class)

    X = []
    Y = []
    Z = []

    x = [xmin + (xmax - xmin) / (width - 1) * i for i in range(width)]
    y = [ymin + (ymax - ymin) / (width - 1) * i for i in range(width)]
    X, Y = np.meshgrid(x, y)
    Z = np.empty((width, width))

    predicted_time = False

    for i in range(width):
        print(f'{i+1}/{width}th iteration')
        for j in tqdm(range(width)):
            if not predicted_time:
                start_time = time.time()
            alfa = x[i]
            beta = y[j]
            vector_i = vector0 + alfa * delta1 + beta * delta2
            vector2model(vector_i, shapes, model_i)

            model_i.eval()
            loss = get_loss(model_i, dataloader, task_type)

            Z[j][i] = loss

            if not predicted_time:
                end_time = time.time()
                elapsed_time = end_time - start_time
                elapsed_time /= 60 # minutes
                all_expected_time = elapsed_time * (width ** 2)
                hour = int(all_expected_time // 60)
                minute = int(all_expected_time - hour * 60)
                print(f'Expected duration time for surface computation: {hour} h {minute} m.')
                predicted_time = True

    return X, Y, Z

def normalize_direction(direction, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.
        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())


def normalize_directions_for_states(direction, states, norm='filter', ignore='ignore'):
    for k in direction:
        if direction[k].dim() <= 1:
            if ignore == 'biasbn':
                direction[k].fill_(0.0) # ignore directions for weights with 1 dimension
            else:
                direction[k].copy_(states[k]) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(direction[k], states[k], norm)


def setup_random_direction(args):
    if args.task_type == "ex-qa":
        model = BertForQuestionAnswering.from_pretrained(args.base_model_path)
    elif args.task_type == 'mc-qa':
        model = AutoModelForMultipleChoice.from_pretrained(args.base_model_path)
    else:
        raise ValueError(f"{args.task_type} is undefined.")

    set_seed(42)
    states = model.state_dict() # a dict of parameters, including BN's running mean/var.
    x_direction = {k: torch.randn(w.size()) for k, w in model.named_parameters()}
    normalize_directions_for_states(x_direction, states, 'filter', 'biasbn')
    y_direction = {k: torch.randn(w.size()) for k, w in model.named_parameters()}
    normalize_directions_for_states(y_direction, states, 'filter', 'biasbn')

    torch.save(x_direction, args.plot_dir / X_DIRECTION)
    torch.save(y_direction, args.plot_dir / Y_DIRECTION)
    print('Saved directions.')

def plot_random_surface(args):
    batch_size = args.batch_size

    set_seed(42)
    if args.task_type == 'ex-qa':
        max_query_length = 64
        max_seq_length = 384
        doc_stride = 128
        tokenizer = BertTokenizer.from_pretrained(args.model_path)
        model0 = BertForQuestionAnswering.from_pretrained(args.model_path)
        model_i = BertForQuestionAnswering.from_pretrained(args.model_path)
        model_i.to(device)
        dataloader = prepare_expa_dataloader(args.task_name, tokenizer, max_seq_length, doc_stride, max_query_length, batch_size)
    elif args.task_type == 'mc-qa':
        max_seq_length = 512
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        model0 = AutoModelForMultipleChoice.from_pretrained(args.model_path)
        model_i = AutoModelForMultipleChoice.from_pretrained(args.model_path)
        model_i.to(device)
        dataloader = prepare_mcqa_dataloader(args.task_name, tokenizer, max_seq_length, batch_size)
    else:
        raise ValueError(f"{args.task_type} is undefined.")

    shapes = OrderedDict()
    pointer = 0
    for k, v in model0.named_parameters():
        if v.requires_grad:
            shapes[k] = {
                'shape': v.shape,
                'start_index': pointer,
            }
            pointer += np.prod(v.shape)

    vector0 = state2vector(model0.state_dict(), shapes)

    x = torch.load(args.plot_dir / X_DIRECTION)
    delta1 = state2vector(x, shapes)
    y = torch.load(args.plot_dir / Y_DIRECTION)
    delta2 = state2vector(y, shapes)

    X, Y, Z = prepare_surface(vector0, delta1, delta2, shapes, model_i, dataloader, width=args.width)

    np.save(args.surface_dir / 'X.npy', X)
    np.save(args.surface_dir / 'Y.npy', Y)
    np.save(args.surface_dir / 'Z.npy', Z)
    print('Saved surface data files')

    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(args.surface_dir / OUTPUT_PDF_FILE, dpi=300,
                bbox_inches='tight', format='pdf')
    print('Saved surface PDF file')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--plot_id",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--surface_id",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=101,
        help="",
    )
    parser.add_argument(
        "--do_setup",
        action='store_true',
        help="",
    )
    parser.add_argument(
        "--do_random_plot",
        action='store_true',
        help="",
    )
    args = parser.parse_args()
    args.plot_dir = Path(os.environ['RE_VISUALIZATION_DIR']) / args.plot_id
    args.plot_dir.mkdir(exist_ok=True)
    args.surface_dir = args.plot_dir / args.surface_id
    args.surface_dir.mkdir(exist_ok=True)
    # triangle()
    if args.do_setup:
        if (args.plot_dir / X_DIRECTION).exists() or (args.plot_dir / Y_DIRECTION).exists():
            print('Direction files already exists.')
        else:
            setup_random_direction(args)
    if args.do_random_plot:
        plot_random_surface(args)
