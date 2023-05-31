# register-qa

Repository for finding questions and answers from texts using register identification. 

The data used to train the register model:

[CORE-corpus](https://github.com/TurkuNLP/CORE-corpus), [FinCORE_full](https://github.com/TurkuNLP/FinCORE_full/releases/tag/v1.0) and [Swedish](https://github.com/TurkuNLP/multilingual-register-data/tree/main/SweCORE/files-with-mt) and [French](https://github.com/TurkuNLP/multilingual-register-data/tree/main/FreCORE/files-with-mt) data from [multilingual-register-data](https://github.com/TurkuNLP/multilingual-register-data).


TODO:

Idea is to at some point map all QA related sublabels under one label, although they may be in different upper level categories. i.e. turn QA into an upper level label and include there QA, FA, FI, FH.

- Take out all MT related ones from SWE and FRE (also from others have to check)!
-- even though there are lots with FA 
- if possible could just use sub_register map and delete examples from the dataset that have the MT label
----> filtering rows of dataset https://huggingface.co/docs/datasets/v1.1.1/processing.html

update sub_register_map in register-multilabel.py!!!!!!
update unique labels as well(?), maybe unnecessary to even have that lol if I use the subcategory mapping anyway :D
add custom loss function for the minority classes, can use the one from toxicity or one anna made before for register stuff as well?


later if multi-label does not work even when mapped under one QA label, try binary (or well still multiclass) stuff where it is QA vs. something else