# Fitness Nutrition Question-Anaswering System

The goal of this project is to generate a Question-Answering (QA) system in Fitness Nutrition.

As a certified Specialist of Fitness Nutrition and a Personal Trainer at ISSA (International Sports Sciences Association), I was trained with two well-designed courses in the aspects of nutrition and training. The context of this project was scraped from Fitness Nutrition book downloads of the certification program: ISSA-Fitness-Nutrition-Certification-Main-Course-Textbook and ISSA-Fitness-Nutrition-Certification-Workbook. After finishing this certificate course, I answered all of self-learnt questions in ISSA-Fitness-Nutrition-Certification-Workbook with guidances from ISSA experts.

Unfortunately, according to ISSA IP policy, the contents in both textbook and workbook are not released to the public.

Due to limited question/answer pairs to build a Fitness Nutrition QA system, I implementate Question Generation Augmentation by using part of the QG pipeline from the "Probably Asked Questions" (PAQ) research from Facebook AI Research (https://github.com/facebookresearch/PAQ).   

Two models are borrowed from PAQ research to generate our question generation augmentation pipeline: 1. Answer Extractor, 2. Question Generator.


## Answer Extraction using PAQ
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

The output of answer extraction is also a jsonl file with the following format which is accepted by the Question Generation component
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
              {"score": -1.7981932736504511, "text": "energy balance", "start": 128, "end": 142}, 
              {"score": -2.287053819110478, "text": "more than weight loss or gain", "start": 43, "end": 72},
              {"score": -2.29739901923343, "text": "body composition", "start": 870, "end": 886}, 
              {"score": -2.3490095799609128, "text": "good nutrition", "start": 19, "end": 33} 
              ]          
  "metadata": {"title": "As I hinted above, good nutrition is about more th", 
               "ps_score": "1"}}
 ```
 
## Question Generation using PAQ
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
The output is a jsonl file with the following format
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
  "passage_id": "ID for passage", 
  "answer": "Benedict", 
  "question": "which pope has the middle name gregory",
  "metadata": {"answer_start": 617, "answer_end": 625, "ae_score": "score for answer", "qg_score": "currently not implemented, but score for question can go here"}
}
```


With subjectively personal generated QA pairs and augmented QA pairs by PAQ, the project uses DistilBERT-based Transformer as baseline and fine-tune on downstream task-specific data (fitness nutrition QA pairs). The current project uses all QA pairs generated by PAQ pipeline. In future's research, I'll use the filtering component of PAQ to improve augmented QA quality.  


