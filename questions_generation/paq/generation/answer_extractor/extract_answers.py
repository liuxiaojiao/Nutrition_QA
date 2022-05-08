#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
from paq.paq_utils import load_jsonl, dump_jsonl, load_dpr_tsv
from paq.generation.answer_extractor.extractors import load_answer_extractor
import logging
import argparse

logger = logging.getLogger(__name__)


# input: a config file and passages (format of the Passage Scoring/Ranking functionality)


# Input: jsonl file with the following format 
# {
#   "passage_id": "ID for passage", 
#   "passage": "Main text of passage.",
#   "metadata": {"title": "Title of passage", "ps_score": "passage score"}
# }


def load_passages(path):
    try:
        return load_jsonl(path)
    except:
        return load_dpr_tsv(path)


def extract_answers(config, input_file, verbose):
    answer_extractor = load_answer_extractor(config)
    passages = load_passages(input_file)
    logger.info("Running answer extractor...")
    annotations = answer_extractor.extract_answers_from_passages(passages, disable_tqdm=not verbose)
    return annotations


def extract_answers_and_write_to_file(config, input_path, output_path, verbose):
    annotations = extract_answers(config, input_path, verbose)
    logger.info('writing extracted answers to file...')
    dump_jsonl(annotations, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extract answers from passages")
    parser.add_argument('--passages_to_extract_from', type=str, required=True, help='path to passages to extract in jsonl format')
    parser.add_argument('--output_path', type=str, required=True, help='Path to dump results to')
    parser.add_argument('--path_to_config', type=str, required=True, help='path to answer extractor config file')
    parser.add_argument('-v', '--verbose', action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    with open(args.path_to_config) as f:
        config = json.load(f)

    if 'answer_extractor' in config:
        config = config['answer_extractor']

    extract_answers_and_write_to_file(config, args.passages_to_extract_from, args.output_path, args.verbose)




# run answer extraction
# python -m paq.generation.answer_extractor.extract_answers \
#     --passages_to_extract_from my_passages_with_scores.jsonl \
#     --output_path my_pasages_with_answers.jsonl \
#     --path_to_config generator_configs/answer_extractor_configs/learnt_answer_extractor_config.json \
#     --verbose


# This will output a jsonl file with the following format (which is accepted by the Question Generation component below)

# {
#   "passage_id": "ID for passage", 
#   "passage": "Main text of passage.",
#   "metadata": {"title": "Title of passage", "ps_score": "passage score"},
#   "answers": [{"text": "Main", "start": 0, "end": 5, "score": "score for answer"}, {"text": "passage", "start": 13, "end": 20, "score": "score for answer"}]
# }

