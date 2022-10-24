"""Defining shortcut and anti-shortcut examples in Extractive QA"""

import re
import os
import numpy as np
import copy
import random
import json

def load_json(file):
    with open(file, 'r') as f:
        d = json.load(f)
    return d

def save_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

from tqdm import tqdm
from collections import Counter
from multiprocessing import Process
import pickle
from pathlib import Path
import argparse
import spacy

from utils_multiple_choice import processors as mc_processors, InputExample
from src.utils.squad import SquadV1Processor, SquadExample
mc_qa_datasets = ['race', 'reclor']
mc_data_to_dir = {
    'race': os.environ['RACE_DIR'],
    'reclor': os.environ['ReClor_DIR']
}

def get_offset_id2w(doc):
    # from char index to word index
    id2w = {}
    tm1 = 0
    for i, t in enumerate(doc):
        for j in range(tm1, t.idx):
            # t.idx: char-level start position of t (token)
            id2w[j] = i - 1
        tm1 = t.idx
    for j in range(t.idx, t.idx+len(t)+1):
        id2w[j] = i
    return id2w

def get_common_ngram(sent1, sent2, n):
    """
    sent1, sent2 (str): lower cased tokens joined with spaces
    n (int): n-gram
    """
    if len(sent1.split()) < n or len(sent2.split()) < n:
        return 0
    m = 0
    tokens = sent2.split()
    for i in range(len(tokens) - n + 1):
        ngram = tokens[i:i+n]
        assert len(ngram) == n
        ngram = ' '.join(ngram)
        if ngram in sent1:
            m += 1
    return m


def main(data_path_or_name, output_dir, n_workers, do_light, do_only_concat, split='train'):
    output_dir = Path(output_dir)
    def _worker(i, data, data_name, do_light, task_type):
        reuslts_dict = analyze(data, do_light, task_type)
        for analysis in analyses:
            save_json(reuslts_dict[analysis], output_dir / f'cache/{analysis}_{data_name}_{i}.json')

    if data_path_or_name in mc_qa_datasets:
        # Multiple-choice QA datasets
        task_type = 'mc-qa'
        processor = mc_processors[data_path_or_name]()
        data_dir = mc_data_to_dir[data_path_or_name]
        if split.lower() == 'train':
            examples = processor.get_train_examples(data_dir)
        elif split.lower() == 'dev':
            examples = processor.get_dev_examples(data_dir)
        elif split.lower() == 'test':
            examples = processor.get_test_examples(data_dir)
        else:
            raise ValueError(f'split should be train, dev or test,'
            ' but your split is {split}.')
        data_name = f'{data_path_or_name}-{split}'
    else:
        # Extractive QA datasets
        task_type = 'ex-qa'
        processor = SquadV1Processor()
        examples = processor.get_train_examples(None, filename=data_path_or_name, do_lower_case=False)
        data_path = Path(data_path_or_name)
        data_name = data_path.stem

    n_works = len(examples) // n_workers
    processes = []

    if not do_only_concat:
        if n_workers > 1:
            for i in range(n_workers):
                if i != n_workers - 1:
                    inputs = examples[i*n_works:(i+1)*n_works]
                else:
                    inputs = examples[i*n_works:]
                process = Process(target=_worker,
                                  args=(i, inputs, data_name, do_light, task_type))
                processes.append(process)
            if len(processes) != 0:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join()
        else:
            reuslts_dict = analyze(data, do_light, task_type)
            for analysis in analyses:
                save_json(reuslts_dict[analysis], output_dir / f'{analysis}_{data_name}.json')
    if n_workers > 1:
        for analysis in analyses:
            results = {}
            for i in tqdm(range(n_workers)):
                file = output_dir / f'cache/{analysis}_{data_name}_{i}.json'
                if os.path.exists(file):
                    result = load_json(file)
                    results.update(result)
                else:
                    print('not exists:', file)
            save_json(results, output_dir / f'{analysis}_{data_name}.json')


def analyze(data, light, task_type):
    results_dict = {analysis: {} for analysis in analyses}
    nlp = spacy.load("en_core_web_sm")
    count = 0
    for example in data:
        count += 1
        if isinstance(example, InputExample):
            assert task_type == 'mc-qa'
            _id = example.example_id
            c = example.contexts[0]
            q = example.question
            if q.find("_") != -1:
                # this is for cloze question
                q = q.replace("_", "").strip()
            q = q.lower()
            options = example.endings
            a = options[int(example.label)]

            del options[int(example.label)]

        elif isinstance(example, SquadExample):
            assert task_type == 'ex-qa'
            _id = example.qas_id
            c = example.context_text
            q = example.question_text
            a = example.answers[0]['text']

            astart = example.answers[0]['answer_start']
            aend = astart + len(a) - 1

        else:
            assert task_type == 'ab-qa'
            raise NotImplementedError()

        try:
            doc = nlp(c)
        except:
            print(f"Parsing context error: {c}")
            continue
        cw = [t.text.lower() for t in doc]
        cs = [sent.text.lower() for sent in doc.sents]

        q_doc = nlp(q)
        qw = [t.text.lower() for t in q_doc]

        if task_type == 'ex-qa':
            id2w = get_offset_id2w(doc)
            aw_start = id2w[astart]
            aw_end = id2w[aend] + 1
            aw = [t.text.lower() for t in doc[aw_start:aw_end]]
        else:
            a_doc = nlp(a)
            aw = [t.text.lower() for t in a_doc]

        if 'answer-position-sentence' in analyses:
            for i, sent in enumerate(doc.sents):
                if sent.start <= aw_start and aw_end <= sent.end:
                    results_dict['answer-position-sentence'][_id] = i
                    break
            if not _id in results_dict['answer-position-sentence']:
                results_dict['answer-position-sentence'][_id] = None

        if 'question-context-ngram-overlap-per-sent' in analyses:
            ngram_overlap_per_sent = []
            q_str = ' '.join([t.text.lower() for t in q_doc])
            for sent in doc.sents:
                sent_str = ' '.join([t.text.lower() for t in sent])
                ngram_overlap = {}
                for n in range(1, 5):
                    # n: ngram, [1, 4]
                    m = get_common_ngram(sent_str, q_str, n)
                    ngram_overlap[f'{n}-gram'] = m
                ngram_overlap_per_sent.append(ngram_overlap)
            results_dict['question-context-ngram-overlap-per-sent'][_id] = ngram_overlap_per_sent

        if 'question-context-similar-sent' in analyses:
            assert 'question-context-ngram-overlap-per-sent' in analyses

            cands = [i for i in range(len(ngram_overlap_per_sent))]
            for n in [4, 3, 2, 1]:
                if len(cands) == 1:
                    break
                ngrams = [ngram_overlap_per_sent[x][f'{n}-gram'] for x in cands]
                max_ngram = max(ngrams)
                cands = list(filter(lambda x: ngram_overlap_per_sent[x][f'{n}-gram'] == max_ngram, cands))
            if len(cands) == 1:
                results_dict['question-context-similar-sent'][_id] = cands[0]
            else:
                results_dict['question-context-similar-sent'][_id] = None

        if 'answer-candidates' in analyses:
            labels = []
            ents = []
            a_ent = ''
            a_is_ent = False
            a_ent_label = None
            for e in doc.ents:
                e_start = e.start
                e_end = e.end
                if aw_end <= e_start:
                    pass
                elif e_end <= aw_start:
                    pass
                else:
                    a_pos = []
                    for w in doc[aw_start:aw_end]:
                        a_pos.append(w.pos_)
                    skip = False
                    if 'VERB' in a_pos:
                        skip = True
                    if 'why' in qw:
                        skip = True
                    if (not skip) and (not a_is_ent):
                        a_is_ent = True
                        a_ent_label = e.label_
                        a_ent = e.text
                labels.append(e.label_)
                ents.append(e.text)

            if a_is_ent:
                assert len(labels) == len(ents)
                ent_with_same_label = []
                for _label, _ent in zip(labels, ents):
                    if _label == a_ent_label:
                        ent_with_same_label.append(_ent)
                n_same_type = len(set(ent_with_same_label))
                results_dict['answer-candidates'][_id] = n_same_type
                if 'answer-entity-type' in analyses:
                    results_dict['answer-entity-type'][_id] = a_ent_label
                if count % 100 == 0:
                    print(f'Entity < {a_ent} > vs. answer < {a} > is matched')
            else:
                assert not _id in results_dict['answer-candidates']
                results_dict['answer-candidates'][_id] = None
                if 'answer-entity-type' in analyses:
                    results_dict['answer-entity-type'][_id] = None
                if count % 100 == 0:
                    print(f'Answer {a} is not matched')

    return results_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path_or_name', type=str, default=None, help='Dataset file path')
    parser.add_argument('--n_workers', type=int, default=1, help='Num workers')
    parser.add_argument('--do_light', action='store_true', help='Do light')
    parser.add_argument('--do_only_concat', action='store_true', help='Do only concat')
    parser.add_argument('--analyses', type=str, nargs='*', default=[], help='Analayze statistics')
    parser.add_argument('--output_dir', type=str, default='output/analysis/ex-qa')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--debug', action='store_true', help='Do debug')
    args = parser.parse_args()
    analyses = args.analyses
    if args.debug:
        raise NotImplementedError
        data = load_json(args.data_path_or_name)['data']
        all_para = []
        for d in data:
            all_para.extend(d['paragraphs'])
        results_dict = analyze(all_para, args.do_light, task_type)
    else:
        main(args.data_path_or_name, args.output_dir, args.n_workers, args.do_light, args.do_only_concat, split=args.split)
    print("Finish!!")
