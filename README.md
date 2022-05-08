# Fitness Nutrition Question-Anaswering System

The goal of this project is to generate a Question-Answering (QA) system in Fitness Nutrition.

As a certified Specialist of Fitness Nutrition and a Personal Trainer at ISSA (International Sports Sciences Association), I was trained with two well-designed courses in the aspects of nutrition and exercise. The context of this project was scraped from the Fitness Nutrition textbook of the certification program: ISSA-Fitness-Nutrition-Certification-Main-Course-Textbook. 

Unfortunately, according to ISSA IP policy, the contents in any textbook are not released to the public. If you request a sample data, you may contact representatives on [ISSA website](https://www.issaonline.com/).

## Highlights of Research Methodology
This Nutrition Question-Answering System takes advantages of pretrained transformers which show State-of-the-Art performance in NLP tasks recently. The research starts from training a baseline model using distilbert-base-uncased, and then fine-tunes on SQuAD pretrained distilbert model. 

I generated 112 QA pairs based on the ISSA Fitness Nutrition textboook. Due to limited question/answer pairs to build a Fitness Nutrition QA system, I implementate Question Generation Augmentation by using part of the QG pipeline from the "Probably Asked Questions" (PAQ) research from [Facebook AI Research](https://github.com/facebookresearch/PAQ). Three models are borrowed from PAQ research to generate our question generation augmentation pipeline: 1. Answer Extractor, 2. Question Generator, and 3. Filtering Generated QA-pairs. Both groudtruth and generated QA paris are used in the training. 

Besides the filtering module in PQA which filters the generated questions for answer consistency, I also use distilbert-base-uncased-distilled-squad (distil bert uncased model pretrained with SQuAD data) from [Huggingface model page](https://huggingface.co/distilbert-base-cased-distilled-squad) to predict the nutrition answers, treat as golden answers and compare with the generated answers to filter out high quality QA pairs for training. 

## Summary of Model Performance
  | Dataset for Training | N size | Model Type | Performance (F1 & EM)
  | --------------- | --------------- | --------------- | --------------- 
  | Ground truth + Generated QA (unfiltered) | 12,097 generated QA & 112 ground truth QA pairs | baseline | F1 = 53.18, EM = 33.55 
  | Ground truth + Generated QA (filtered by Consistent=True) | 4,441 generated QA & 112 ground truth QA pairs | baseline | F1 = 64.86, EM = 52.01
  | Ground truth + Generated QA (unfiltered) | 12,097 generated QA & 112 ground truth QA pairs | fine-tune with SQuAD 1.1 | F1 = 57.55, EM = 38.57
  | Ground truth + Generated QA (filtered by Consistent=True) | 4,441 generated QA & 112 ground truth QA pairs | fine-tune with SQuAD 1.1 | F1 = 81.13, EM = 66.70

_Note: Consistent=True is filtered by PAQ Filtering Generated QA-pairs module._ 
  
Fine-tuning on fitness nutrition data with SQuAD pretrained model/weights significantly improves the model performance. Therefore, I use distilbert-base-uncased-distilled-squad pretrained results to compute the F1 score against all PAQ generated questions as the filtering metric. The F1 filtering thresholds are 1.0, 0.8, 0.6, 0.4. Ground truth pairs are considered with F1 = 1.0.

  | Dataset for Training | N size | Performance (F1 & EM)
  | --------------- | --------------- | --------------- 
  | **Ground truth + Generated QA (filtered by F1=1.0)** | 4,127 generated QA & 112 ground truth QA pairs | **F1 = 71.40, EM = 61.67** 
  | **Ground truth + Generated QA (filtered by F1=0.8)** | 4,943 generated QA & 112 ground truth QA pairs | **F1 = 71.16, EM = 54.83**
  | Ground truth + Generated QA (filtered by F1=0.6) | 6,256 generated QA & 112 ground truth QA pairs | F1 = 69.55, EM = 49.61
  | Ground truth + Generated QA (filtered by F1=0.4) | 7,903 generated QA & 112 ground truth QA pairs | F1 = 63.91, EM = 41.75

Based on the above experiments, ***implement PAQ for data augmentation and filter good quality QA paris based on pretrained SQaUD transformers are effective in improving model performance in the few-shot downstream NLP tasks.*** Although F1 threshold is one of the hyperparameters to tune, F1 threshold = 0.8 may garantee satisfacotry quality of question generation with relatively good amount of QA pairs.

## Future Research
- Improve the data augmentation and the filtering metrics to improve augmented QA quality, as this is important for NLP downstream few-shot learning. 
- Expand the context topics to areas of exercise, nutrition and dietetics, chronic disease management, psychological and physical health management.
- Explore fitness nutrition textbook to generate more good quality of ground truth QA pairs for training.



## PAQ for Question Generation 
################################ TL;DR ################################

Refer to [PAQ](https://github.com/facebookresearch/PAQ) from facebook research. Brief implementation process below.

### 1. Answer Extractor
The learnt answer span extractor model, answer_extractor_nq_base (Learnt Answer Span Extractor, BERT-base, NQ-trained), is used to perform the answer extraction on passages, which is a BERT-base architecture that has been fine-tuned on NQ dataset.

Below is an example to extract answers from passages, using the learnt extractor:

```
# download the span extractor model:
python -m paq.download -v -n models.answer_extractors.answer_extractor_nq_base

# run answer extraction
python -m paq.generation.answer_extractor.extract_answers \
    --passages_to_extract_from my_passages.jsonl \
    --output_path my_pasages_with_answers.jsonl \
    --path_to_config generator_configs/answer_extractor_configs/learnt_answer_extractor_config.json \
    --verbose
```

The input of Answer Extraction, my_passages.jsonl, was cleaned and saved as a jsonl file with the following format which is accepted by the Answer Extraction component.
```
{
  "passage_id": "ID for passage", 
  "passage": "Main text of passage.",
  "metadata": {"title": "Title of passage", "ps_score": "passage score"}
}
```

The output of answer extraction is also a jsonl file with the following format which is accepted by the Question Generation component:
```
{
  "passage_id": "ID for passage", 
  "passage": "Main text of passage.",
  "metadata": {"title": "Title of passage", "ps_score": "passage score"},
  "answers": [{"text": "Main", "start": 0, "end": 5, "score": "score for answer"}, {"text": "passage", "start": 13, "end": 20, "score": "score for answer"}]
}
```
An example of output of answer extraction is:
```
{
  "passage_id": 32, 
  "passage": "As I hinted above, good nutrition is about more than weight loss or gain. ......
              performance at the elite, world-class level; it all depends on your client\u2019s goals and activities.", 
  "answers": [
              {"score": -1.798, "text": "energy balance", "start": 128, "end": 142}, 
              {"score": -2.287, "text": "more than weight loss or gain", "start": 43, "end": 72},
              {"score": -2.297, "text": "body composition", "start": 870, "end": 886}, 
              {"score": -2.349, "text": "good nutrition", "start": 19, "end": 33} 
              ]          
  "metadata": {"title": "As I hinted above, good nutrition is about more th", 
               "ps_score": "1"}}
 ```
 
### 2. Question Generator
The BART-base model trained on NQ and Multitask datasets is used to perform the question generation based on ansers extracted from last step. 

Below is an example to generate questions from passages with extracted answers, using the multitask generator:

```
# download the qgen model:
python -m paq.download -v -n models.qgen.qgen_multi_base

# run question generation extraction
python -m paq.generation.question_generator.generate_questions \
    --passage_answer_pairs_to_generate_from my_pasages_with_answers.jsonl \
    --output_path my_generated_questions.jsonl \
    --path_to_config generator_configs/question_generator_configs/question_generation_config.json \
    --verbose
```
The output is a jsonl file with the following format:
```
{
  "passage_id": "ID for passage", 
  "answer": "Benedict", 
  "question": "which pope has the middle name gregory",
  "metadata": {"answer_start": 617, "answer_end": 625, "ae_score": "score for answer", "qg_score": "currently not implemented, but score for question can go here"}
}
```
An example of output of question generation is:
```
{
  "passage_id": 32, 
  "answer": "good nutrition", 
  "question": "what is more important than weight loss or gain", 
  "metadata": {"answer_start": 19, "answer_end": 33, "ae_score": -2.35, "qg_score": null}
}
```
### 3. Filtering Generated QA-Pairs
Filter the generated questions for answer consistency based on "DPR Passage retriever and faiss index" (BERT-base) and "FID-base reader" (t5-base).

Below is an example of filtering generated questions with local filtering (faster one, global filtering takes longer but with better performance):
```
python -m paq.generation.filtering.filter_questions \
    --generated_questions_to_filter my_generated_questions.jsonl \
    --output_path my_locally_filtered_questions.jsonl \
    --path_to_config generator_configs/filterer_configs/local_filtering_config.json \
    --verbose
```    
The output is a jsonl file with the following format:
```
{
  "passage_id": "ID for passage", 
  "answer": "Benedict", 
  "question": "which pope has the middle name gregory",
  "metadata": {"filter_answer": "benedict", "consistent": true, "answer_start": 617, "answer_end": 625, "ae_score": "score for answer", "qg_score": "currently not implemented, but score for question can go here"}
}
```

