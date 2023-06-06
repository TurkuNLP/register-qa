# register-qa

Repository for finding questions and answers from texts using register identification. 

The data used to train the register model:

[CORE-corpus](https://github.com/TurkuNLP/CORE-corpus), [FinCORE_full](https://github.com/TurkuNLP/FinCORE_full/releases/tag/v1.0) and [Swedish](https://github.com/TurkuNLP/multilingual-register-data/tree/main/SweCORE/files-with-mt) and [French](https://github.com/TurkuNLP/multilingual-register-data/tree/main/FreCORE/files-with-mt) data from [multilingual-register-data](https://github.com/TurkuNLP/multilingual-register-data).

Map all QA related sublabels under one label, although they may be in different upper level categories. i.e. turn QA into an upper level label and include there QA, FA, FI, FH. This is done via sub_register_map in the script. Use upper labels with QA_NEW.

TODO:

- comet_ml automatic logging, figure out why it creates random name and then "checkpoints" (trainer output dir) where all the metrics etc. go. Works for training though so good enough for now.
- QA_NEW gets 0.65 with base and class weights, maybe put more importance to that class still (don't know how I would go about that yet with the method I am doing)? Anyway the result is decent though I think

- Get document start and document end or something, label for each part?

- had problems with the dataset and tokenization, text field was missing for some english data (train at least)
-> IN DT but no text, line 95, also 301 same problem, 6436
-> ID QA 22701
-> ID DF 11815
-> NA NE 13800
-> SP OS 20585
-> IN 14130
THESE ATLEAST, NOW I FILTER THEM AWAY IF THERE IS NO TEXT FIELD IN THE CODE
-> mention to Veronika, I might have had this problem already last summer(?)


- later if multi-label does not work even when mapped under one QA label, try binary (or well still multiclass) stuff where it is QA vs. something else