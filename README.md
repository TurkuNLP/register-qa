# register-qa

Repository for finding questions and answers from texts using register identification as well as for splitting texts to questions and answers using multiple methods such as rule-based things, token classification and other methods.

The data used to train the register model:

[CORE-corpus](https://github.com/TurkuNLP/CORE-corpus), [FinCORE_full](https://github.com/TurkuNLP/FinCORE_full/releases/tag/v1.0) and [Swedish](https://github.com/TurkuNLP/multilingual-register-data/tree/main/SweCORE/files-with-mt) and [French](https://github.com/TurkuNLP/multilingual-register-data/tree/main/FreCORE/files-with-mt) data from [multilingual-register-data](https://github.com/TurkuNLP/multilingual-register-data).

Map all QA related sublabels under one label, although they may be in different upper level categories. i.e. turn QA into an upper level label and include there QA, FA, FI, FH. This is done via sub_register_map in the script. Use upper labels with QA_NEW.

<!-- TODO:

- comet_ml automatic logging, figure out why it creates random name and then "checkpoints" (trainer output dir) where all the metrics etc. go. Works for training though so good enough for now.

- had problems with the dataset and tokenization, text field was missing for some english data (train at least) 
NOW I FILTER THEM AWAY IF THERE IS NO TEXT FIELD IN THE CODE
-> mention to Veronika, I might have had this problem already last summer(?) -->