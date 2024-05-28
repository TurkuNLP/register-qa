# register-qa

Repository for finding questions and answers from texts using register identification as well as for splitting texts to questions and answers using token classification.

The repository includes the scripts, logs, and some of the data as well.

![image](https://github.com/TurkuNLP/register-qa/assets/92675015/39b67d70-630e-46e7-8308-514732e0d3f0)

![image](https://github.com/TurkuNLP/register-qa/assets/92675015/953199c8-1057-4b31-9ad5-68bb22604bd4)

## Citing
To cite our work please use the following bibtex.

```
@inproceedings{eskelinen-etal-2024-building-question,
    title = "Building Question-Answer Data Using Web Register Identification",
    author = "Eskelinen, Anni  and
      Myntti, Amanda  and
      Henriksson, Erik  and
      Pyysalo, Sampo  and
      Laippala, Veronika",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.234",
    pages = "2595--2611",
    abstract = "This article introduces a resource-efficient method for developing question-answer (QA) datasets by extracting QA pairs from web-scale data using machine learning (ML). Our method benefits from recent advances in web register (genre) identification and consists of two ML steps with an additional post-processing step. First, using XLM-R and the multilingual CORE web register corpus series with categories such as QA Forum, we train a multilingual classifier to retrieve documents that are likely to contain QA pairs from web-scale data. Second, we develop a NER-style token classifier to identify the QA text spans within these documents. To this end, we experiment with training on a semi-synthetic dataset built on top of the English LFQA, a small set of manually cleaned web QA pairs in English and Finnish, and a Finnish web QA pair dataset cleaned using ChatGPT. The evaluation of our pipeline demonstrates its capability to efficiently retrieve a substantial volume of QA pairs. While the approach is adaptable to any language given the availability of language models and extensive web data, we showcase its efficiency in English and Finnish, developing the first open, non-synthetic and non-machine translated QA dataset for Finnish {--} Turku WebQA {--} comprising over 200,000 QA pairs.",
}
```
