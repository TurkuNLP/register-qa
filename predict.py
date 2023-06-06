# import transformers
# import datasets
# from pprint import pprint


# # with pipeline

# model = transformers.AutoModelForSequenceClassification.from_pretrained("") # load model from local directory
# tokenizer = transformers.AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")


# test_pipe = transformers.pipeline(task="text-classification", model=model, tokenizer=tokenizer, function_to_apply="sigmoid", top_k=None) # return_all_scores=True is deprecated

# test = [""] # add examples to test

# results = test_pipe(test)

# for zipped in zip(test, results):
#   pprint(zipped)



import transformers
import torch
import numpy as np
import argparse
from pprint import PrettyPrinter
import json
import datasets
import pandas as pd

""" This script is meant for looking at multi-label predictions for raw text data and saving the probabilities with id. """

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
    help="the model name")
parser.add_argument('--data', required=True,
    help="the file name of the raw text to use for predictions")
parser.add_argument('--tokenizer', required=True,
    help="the tokenizer to use for tokenizing new text")
parser.add_argument('--filename', required=True,
    help="the file name to give file resulting from the predictions")
args = parser.parse_args()
print(args)

pprint = PrettyPrinter(compact=True).pprint


# read the data in
data = args.data

if ".json" in data:
    with open(data, 'r') as json_file:
            json_list = list(json_file)
    lines = [json.loads(jline) for jline in json_list]

    # use pandas to look at each column
    df=pd.DataFrame(lines)

# TODO might have to change this depending on the data type
elif ".tsv" in data:
    with open(data, "rt", encoding="utf-8") as f:
        lines = f.readlines()
    lines = lines[1:]
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")
        lines[i] = lines[i].split("\t")
        assert len(lines[i]) == 3

    df=pd.DataFrame(lines, columns = ['id', 'label', 'text'])


# instantiate model, this is pretty simple
model=transformers.AutoModelForSequenceClassification.from_pretrained(args.model)

trainer = transformers.Trainer(
    model=model
) 

tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

def tokenize(example):
    return tokenizer(
        example["text"],
        padding='max_length', # this got it to work, data_collator could have helped as well?
        max_length=512,
        truncation=True,
    )

dataset = datasets.Dataset.from_pandas(df)

#map all the examples
dataset = dataset.map(tokenize)

texts = dataset["text"]
ids = dataset["id"]

# see how the labels are predicted
test_pred = trainer.predict(dataset)
predictions = test_pred.predictions # these are the logits

sigmoid = torch.nn.Sigmoid()
probs = sigmoid(torch.Tensor(predictions))
probs = probs.tolist()

unique_labels = ["IN", "NA", "HI", "LY", "IP", "SP", "ID", "OP", "QA_NEW"] # upper labels plus qa_new
for index, label in enumerate(unique_labels):
    # get probabilities to their own lists for use in tuple and then dataframe (each with their own column)
    label = [probs[i][unique_labels[index]] for i in range(len(probs))]

# TODO some easy way to unpack these labels into the variables by same name? I guess not
all = tuple(zip(ids, IN, NA, HI, LY, IP, SP, ID, OP, QA_NEW))


allpredict = [item for item in all]

print(*unique_labels)

# get to dataframe
def id_and_label(data):
    df = pd.DataFrame(data, columns=['id', *unique_labels])
    return df

# id and labels + their probabilities
all_dataframe = id_and_label(allpredict)

# put to tsv 
filename = args.filename
all_dataframe.to_csv(filename, sep="\t", index=False)