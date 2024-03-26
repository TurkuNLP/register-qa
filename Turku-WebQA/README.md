# Turku WebQA dataset

Finnish pairs available at [Huggingface Hub](https://huggingface.co/datasets/TurkuNLP/Turku-WebQA).

## Formats

The data, for Finnish and English, is given in two formats. Files are named based on the source (i.e. Falcon, Parsebank, mC4-FI or CC-FI)

Full-extractions (as jsonl):
  - id: id of the document
  - text: original text
  - extracted: a dictionary containing the extracted questions and answers (as well as the non-tagged surrounding text)

Pairs (as .tsv):
  - field 1: id of the document
  - field 2: extracted question
  - field 3: extracted answer
