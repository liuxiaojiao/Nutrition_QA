import argparse
import json
import os
import logging
import string
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import random
from collections import Counter, OrderedDict, defaultdict as ddict
from itertools import compress
random.seed(0)

logger = logging.getLogger(__name__)

def combine_generated_original(generated_df, original_df):
    for i in range(len(generated_df)):
     for j in range(len(original_df)):
        if generated_df[i]['passage_id'] == original_df[j]['passage_id']:
            generated_df[i].update({
                'context': original_df[j]['passage'],
                'title': original_df[j]['metadata']['title']
                })
    return generated_df

def combined_formatted(combined_df_full):
    random.seed(0)
    filtered_df = [k for k in combined_df_full]  

    id_filter = []
    for k in filtered_df:
        id_filter.append(k['passage_id'])  

    # generate qas list of dict
    qas = []
    tmp = [{
            'question': filtered_df[0]['question'],
            'id': ''.join(random.choice(string.ascii_lowercase) for i in range(8)),
            'answers':[{
                'answer_start': filtered_df[0]['metadata']['answer_start'],
                    'text': filtered_df[0]['metadata']['filter_answer']
                    }]
        }]

    for i in range(1,len(filtered_df)):       
        if id_filter[i] == id_filter[i-1]:
            tmp.append({
                        'question': filtered_df[i]['question'],
                        'id': ''.join(random.choice(string.ascii_lowercase) for i in range(8)),
                        'answers':[{
                            'answer_start': filtered_df[i]['metadata']['answer_start'],
                            'text': filtered_df[i]['metadata']['filter_answer']
                        }]
                       })

        if id_filter[i] != id_filter[i-1]:
            if len(tmp) > 0: 
                qas.append(tmp)
                tmp = []

            new_tmp = [{
            'question': filtered_df[i]['question'],
            'id': ''.join(random.choice(string.ascii_lowercase) for i in range(8)),
            'answers':[{
                'answer_start': filtered_df[i]['metadata']['answer_start'],
                    'text': filtered_df[i]['metadata']['filter_answer']
                    }]
            }]
            tmp = new_tmp

        if i == len(id_filter)-1:
            qas.append(tmp)
        
    # generate the full list of dict
    filtered_df_dict = {
            'title': filtered_df[0]['title'],
            'paragraphs': [{
                'context': filtered_df[0]['context'],
                'qas': qas[0]
            }]
        }

    res = [filtered_df_dict]
    filtered_df_index = 0
    for i in range(1,len(filtered_df)):
        if id_filter[i] != id_filter[i-1]:
            filtered_df_index += 1
            res.append(
                {
                'title': filtered_df[i]['title'],
                'paragraphs': [{
                            'context': filtered_df[i]['context'],
                            'qas': qas[filtered_df_index]
                              }]
                }
            )

    return res

def dftable_formated(df):
    random.seed(0)
    df['title'] = df['text'].apply(lambda x: x[:50])
    df['question_id'] = df['question'].apply(lambda x: ''.join(random.choice(string.ascii_lowercase) for i in range(8)))
    df['answer_start'] = df.apply(lambda x: x['text'].rfind(x['answer']), axis=1)

    ground_truth_list = [] 
    for _, row in df.iterrows():
        ground_truth_list.append(
                {'passage_id': row['id'], 
                 'answers': row['answer'], 
                'question': row['question'], 
                'metadata': {'answer_start': row['answer_start'], 'filter_answer': row['answer'], 'consistent': True}, 
                'context': row['text'], 
                'title': row['title']},
        )

    res = combined_formatted(ground_truth_list)

    for m in res:
        for n in m['paragraphs'][0]['qas']:
            n['f1-score'] = 1

    return res

def filter_f1_score(full_list):  
    for i in range(len(full_list)): 
        tmp = []
        for j in range(len(full_list[i]['paragraphs'][0]['qas'])):
            # fine-tune f1-score here: 1, 0.8, 0.6, 0.4
            if full_list[i]['paragraphs'][0]['qas'][j]['f1-score'] < 1:
                tmp.append(False)
            else:
                tmp.append(True)    
          
        full_list[i]['paragraphs'][0]['qas'] = list(compress(full_list[i]['paragraphs'][0]['qas'],tmp))   
    return full_list

def extend_qas(list1, list2):

    for m in list1:
        for n in list2:
            if m['title'] == n['title']:
                m['paragraphs'][0]['qas'].extend(n['paragraphs'][0]['qas'])

    # fine-tune f1-score here: 1, 0.8, 0.6, 0.4
    filtered_list1 = filter_f1_score(list1)  
    return filtered_list1

def count_qa_pairs(formatted_list):
    count = 1
    for k in formatted_list:
        count += len(k['paragraphs'][0]['qas'])
    return count

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def normalize_answer(s):
    """Convert to lowercase and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def append_f1(generated_qa,pred_dict):
    for k in generated_qa:
        for m in k['paragraphs'][0]['qas']:
            m['f1-score'] = compute_f1(m['answers'][0]['text'], pred_dict[m['id']])
    return generated_qa

def main(input_path_generated_full,input_path_predict,input_path_groundtruth,output_path,verbose):

    squad_data_generated = {"data": [], "version": "1.1"}
    training_data_generated = {"data": [], "version": "1.1"}
    validation_data_generated = {"data": [], "version": "1.1"}

    logger.info(f"Loading data from {input_path_generated_full}")
    num_questions = 0.0

    generated_qa_full_formatted = read_file(input_path_generated_full, verbose)
    generated_qa = generated_qa_full_formatted[0]['data']

    predicted_qa = read_file(input_path_predict, verbose)
    pred_dict = dict(zip(predicted_qa['Id'], predicted_qa['Predicted']))

    # generated_df_formatted: cleaned squad data with f1-score
    generated_df_formatted = append_f1(generated_qa, pred_dict)
    generated_df_qa_pairs = count_qa_pairs(generated_df_formatted)

    # add ground truth QA pairs
    ground_truth_df = read_file(input_path_groundtruth, verbose)
    ground_truth_formatted = dftable_formated(ground_truth_df)

    ground_truth_qa_pairs = count_qa_pairs(ground_truth_formatted)

    print('There are {} generated QA & {} ground truth QA pairs originally'.format( \
           generated_df_qa_pairs, ground_truth_qa_pairs))


    final_df_formatted = extend_qas(generated_df_formatted, ground_truth_formatted)
    final_df_qa_pairs = count_qa_pairs(final_df_formatted)
    print('After filtering, there are overall {} QA pairs for training'.format(final_df_qa_pairs))

    squad_data_generated['data'] = final_df_formatted

    with open(os.path.join(output_path, "cleaned_data_squad_filtered.json"), 'w') as fp:
        json.dump(squad_data_generated, fp)

    # dump training (80%) and validation (20%) datasets to json files
    training_data = random.sample(final_df_formatted, k=round(len(final_df_formatted) * 0.8))
    validation_data = []

    for element in final_df_formatted:
        if element not in training_data :
            validation_data.append(element)

    print('{} records of training records'.format(len(training_data)))
    print('{} records of validation records'.format(len(validation_data)))
    
    training_data_generated['data'] = training_data
    validation_data_generated['data'] = validation_data

    with open(os.path.join(output_path, "nutrition_filtered_training.json"), 'w') as fp:
        json.dump(training_data_generated, fp)

    with open(os.path.join(output_path, "nutrition_filtered_validation.json"), 'w') as fp:
        json.dump(validation_data_generated, fp)

def read_file(input_path, verbose):
    if input_path.split('.')[-1] == 'json':
        data = load_json(input_path)

    elif input_path.split('.')[-1] in ['csv','tsv']:
        data = pd.read_csv(input_path, encoding='latin-1')

    print('Loaded {} records from {}'.format(len(data), input_path))
        
    return data

def load_json(input_path):
    """
    Read list of objects from a JSON file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data

if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s " "- %(name)s - %(message)s", level=logging.INFO
    )
    parser = argparse.ArgumentParser(
        description=("Convert a PAQ dataset into SQuADv1.1 format."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_path_generated_full", type=str, required=True, 
        help=("Path to full generated PAQ data to convert.")
    )
    
    parser.add_argument(
        "--input_path_groundtruth", type=str, required=True, 
        help=("Path to groud truth PAQ data to convert.")
    )

    
    parser.add_argument(
        "--input_path_predict", type=str, required=True, 
        help=("Path to predicted answer for filtering.")
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help=("Path prefix to write SQuADv1.1-formatted dataset."),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print warnings when detected answer "
            "doesn't match paragraph-extracted answer."
        ),
    )
    args = parser.parse_args()
    main(args.input_path_generated_full,args.input_path_predict,args.input_path_groundtruth,args.output_path,args.verbose)

