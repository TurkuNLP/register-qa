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
import csv

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

# # TODO might have to change this depending on the data type
# elif ".tsv" in data:
#     with open(data, "rt", encoding="utf-8") as f:
#         lines = f.readlines()
#     lines = lines[1:]
#     for i in range(len(lines)):
#         lines[i] = lines[i].replace("\n", "")
#         lines[i] = lines[i].split("\t")
#         assert len(lines[i]) == 3

#     df=pd.DataFrame(lines, columns = ['id', 'label', 'text'])


elif ".tsv" in data:
    with open(data, "rt", encoding="utf-8") as f:
        lines = f.readlines()
    lines = lines[1:]
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")
        lines[i] = lines[i].split("\t")
        assert len(lines[i]) == 2

    df=pd.DataFrame(lines, columns = ['label', 'text'])


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

labels = dataset["label"]
# oh right I would have to change the labels for the test set to match the upper ones if I wanted easily readable results

dataset = dataset.remove_columns("label")
texts = dataset["text"]
#ids = dataset["id"]

# see how the labels are predicted
test_pred = trainer.predict(dataset)
predictions = test_pred.predictions # these are the logits

sigmoid = torch.nn.Sigmoid()
probs = sigmoid(torch.Tensor(predictions))
probs = probs.numpy()

unique_labels = ["IN", "NA", "HI", "LY", "IP", "SP", "ID", "OP", "QA_NEW"] # upper labels plus qa_new

with open(args.filename, 'w') as outfile:
    header = ["text", "gold_labels", *unique_labels] #maybe I should put the text last
    
    writer = csv.writer(outfile, delimiter="\t")
    writer.writerow(header)

    for i in range(len(texts)):
            
        text = texts[i]
        gold = labels[i]
        line = [text, gold]
        pred_list = [str(val) for val in probs[i]]
        line = [*line, *pred_list]

        writer.writerow(line)