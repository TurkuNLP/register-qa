import transformers
import torch
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import gzip
import json
from torch.nn import DataParallel
import csv
from tqdm import tqdm
import os


DEVICE = "cuda"
DEVICE_COUNT = torch.cuda.device_count()

def log(s):
    print(s, flush=True)

def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--tokenizer_path', 
                    help='Pretrained model name')
    ap.add_argument('--batch_size_per_device', default=32, type=int, help="batch_size per gpu")
    ap.add_argument('--text', metavar='FILE', required=True,
                    help='Text to be predicted') #could also be string?
    ap.add_argument('--model_path', metavar='FILE',
                    help='Load model from file')
    ap.add_argument('--output', default="output_predictions.tsv", metavar='FILE', help='Location to save predictions')
    return ap

def tokenize(batch, tokenizer):
    tokenized = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
    return tokenized

def predict_labels(data, model, options):
    with torch.no_grad():
        logits = model(**data).logits

    predictions = [[logit.argmax(-1)] for logit in logits] # This is what Amanda needs
    #predictions = [[torch.argmax(logit, dim=1)] for logit in logits] # I don't understand dimensions :'D in example on huggingface it is 2, here does not work but -1 at least works, 1 could work?
    predicted_token_class = [[model.config.id2label[t.item()] for t in prediction[0]] for prediction in predictions]
    
    return predicted_token_class


def load_text_batch(handle, options):
    ids = []
    batch = [handle.readline() for _ in range(0, options.batch_size)]
    batch = [line for line in batch if len(line) > 0]
    batch = batch[1:]
    for i in range(len(batch)):
        batch[i] = batch[i].replace("\n", "")
        batch[i] = batch[i].split("\t")
    texts = [row[2] for row in batch]

    ids = [row[0] for row in batch]

    return ids, texts


def main():
    options = argparser().parse_args(sys.argv[1:])
    options.batch_size = options.batch_size_per_device * DEVICE_COUNT
        # tokenizer from the pretrained model
    log("Loading tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(options.tokenizer_path)
    log("...done") 
    # load our fine-tuned model
    log("Loading model")
    # model = torch.load(options.model_path, map_location=torch.device(DEVICE))
    model = transformers.AutoModelForTokenClassification.from_pretrained(options.model_path)
    model.to(DEVICE)


    with gzip.open(options.text, 'rt') as f:
        with open(options.output, 'w') as outfile:
            header = ["id", "labels", "text"]

            writer = csv.writer(outfile, delimiter="\t")
            writer.writerow(header)
            
            ids, text_batch = load_text_batch(f, options)

            while text_batch:

                tokenized_batch = tokenize(text_batch, tokenizer).to(DEVICE)
                labels = predict_labels(tokenized_batch, model, options) 

                for sentence, labelled in zip(text_batch, labels):
                    sentence = sentence.split()
                    for word, label in zip(sentence, labelled):
                        print(word, label)

                for i in range(len(text_batch)):
                    
                    # TODO this have to figure out how to best save :D
                    text = text_batch[i].replace("\n", "\\n")
                    line = [ ids[i],  " ".join(labels[i]), text ] # this has to be something else :D

                    writer.writerow(line)
                
                ids, text_batch = load_text_batch(f, options)

if __name__ == "__main__":
    main()