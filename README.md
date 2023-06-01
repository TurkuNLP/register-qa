# register-qa

Repository for finding questions and answers from texts using register identification. 

The data used to train the register model:

[CORE-corpus](https://github.com/TurkuNLP/CORE-corpus), [FinCORE_full](https://github.com/TurkuNLP/FinCORE_full/releases/tag/v1.0) and [Swedish](https://github.com/TurkuNLP/multilingual-register-data/tree/main/SweCORE/files-with-mt) and [French](https://github.com/TurkuNLP/multilingual-register-data/tree/main/FreCORE/files-with-mt) data from [multilingual-register-data](https://github.com/TurkuNLP/multilingual-register-data).

Map all QA related sublabels under one label, although they may be in different upper level categories. i.e. turn QA into an upper level label and include there QA, FA, FI, FH. This is done via sub_register_map in the script.

TODO:

- check with veronika the sub_register_map and unique labels, do we have to change anything other than the qa related things
- add class weights for the minority classes, can use the one from toxicity or one anna made before for register stuff as well?
- add use of some automatic logging thing, could not get the one I tried with toxicity working so try something else? > DONE? comet-ml has automatic logging I guesss, if it does not work I coult try weights and biases


- having problems with the dataset and tokenization, weird stuff, maybe need to check that each line is working fine (as in does not have more or less columns than it should)

later if multi-label does not work even when mapped under one QA label, try binary (or well still multiclass) stuff where it is QA vs. something else