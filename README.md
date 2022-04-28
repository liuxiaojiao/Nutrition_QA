# Fitness Nutrition Question-Anaswering System

The goal of this project is to generate a Question-Answering (QA) system in Fitness Nutrition.

As a certified Specialist of Fitness Nutrition. The context of this project was scraped from ISSA 

After finishing the certificate course, I generated the question/answer pairs

Due to limited question/answer pairs for training, I implementate Question Generation Augmentation by using part of the QG pipeline from the "Probably Asked Questions" (PAQ) research from Facebook AI Research (https://github.com/facebookresearch/PAQ).   


## Answer Extraction using PAQ
To perform answer extraction on passages, 

The learnt answer span extractor model, answer_extractor_nq_base (Learnt Answer Span Extractor, BERT-base, NQ-trained), is based on BERT-base architecture and fine-tuning on NQ dataset.

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


