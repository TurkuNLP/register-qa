import transformers
import torch
from datasets import load_from_disk
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import gzip
import json
#from torch.nn import DataParallel
import csv
#from tqdm import tqdm
import os
from difflib import SequenceMatcher
import re


DEVICE = "cuda"
DEVICE_COUNT = torch.cuda.device_count()
USE_SEQ_MATCHER=True
CUTOFF_FOR_SQM=0.6
MAX_LENGTH = 512


# FOR FINAL EVALUATION
MODEL_PREFIX = "/scratch/project_2005092/Anni/qa-register/models_for_Amanda/"
#DATA_EN = "/scratch/project_2005092/Anni/qa-register/token-qa/datasets/dataset_punct2/"
DATA_EN = "/scratch/project_2005092/Anni/qa-register/token-qa/samples/sample-falcon/annotated/formatted/en_test_formatted.jsonl"
DATA_FI = "/scratch/project_2002026/amanda/register-qa/data/test_formatted.jsonl"

def log(s):
    print(s, flush=True)

def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    #ap.add_argument('--batch_size_per_device', default=32, type=int, help="batch_size per gpu")
    ap.add_argument('--input', metavar='FILE',
                    help='File to be predicted')
    ap.add_argument('--input_lang', required=True, choices=['en','fi','detect'],
                    help="Chooses which data to predict")
    ap.add_argument('--input_format', default="dataset",
                    help='type of the input, dataset object or zipped (only dataset works!)')
    ap.add_argument('--remove_special_tokens', default=True,
                    help='remove special tokens in the tokenisation process')
    ap.add_argument('--label_start_only', default=False,
                    help='only label start of question, answer, inside = O')
    ap.add_argument('--input_key', default="validation",
                    help='which split to read from a dataset object') #could also be string?
    ap.add_argument('--model_path', metavar='FILE', 
                    default="",  #/scratch/project_2005092/Anni/qa-register/models/token-qa/fin_and_gtp_model
                    help='Load model from file')
    ap.add_argument('--tokenizer_path', default="xlm-roberta-base",
                    help='Pretrained model name, to download the tokenizer')
    ap.add_argument('--output', default="", metavar='FILE', help='Location to save predictions')
    return ap

id2label= {
    0: "Q",
    1: "A",
    2: "O",
    -100: "O"}
label2id = {"Q":0,
            "A":1,
            "O":2}


#def tokenize(batch, tokenizer):
#    tokenized = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
#    return tokenized

def predict_labels(data, model, options):
    with torch.no_grad():
        logits = model(**data).logits
    #print(logits.size())
    predictions = [[logit.argmax(-1)] for logit in logits] # This is what Amanda needs
    #predicted_token_class = [[model.config.id2label[t.item()] for t in prediction[0]] for prediction in predictions]
    #predicted_token_class = [[id2label[t.item()] for t in prediction[0]] for prediction in predictions]
    predicted_token_class = [[t.item() for t in prediction[0]] for prediction in predictions]
    #print(predicted_token_class[0])
    
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


def tokenize_and_align_labels(example, tokenizer, options):
    """
    Tokenizes and rearranges the labels, i.e.
    [Hi, is, this, yours?]  => [CLS, "Hi", ",", "is", "this", "yours", "?", CLS] 
    [ O,  Q,    Q,      Q]     [  O,   O,   O,    Q,     Q,       Q,    Q,   O ]
    Special tokens mapped to "O"==empty later.
    """
    # adapted from https://huggingface.co/docs/transformers/custom_datasets#tok_ner

    tokenized = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
        return_tensors='pt')
    tags = example['tags']
    word_ids = tokenized.word_ids()
    labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        # Special tokens have word id None. Set their label to -100 so they
        # are automatically ignored in the loss function.
        if word_idx is None:
            labels.append(-100)
        # Set the label for the first token of each word normally.
        elif word_idx != previous_word_idx:
            labels.append(tags[word_idx])
        # For subsequent tokens in a word, set the label to -100.
        else:
            if options.label_start_only:
                labels.append(-100)
            else:
                labels.append(labels[-1])
        previous_word_idx = word_idx
    #tokenized["labels"] = labels
    return tokenized, labels



#................................For reading the annotation format......................................#

def flatten(l):
    return [item for subl in l for item in subl]


def find_sub_list(sl,l):
    """ From stackexchange, find indices of a sublist in a list (seqmatch added) """
    sll=len(sl)
    if USE_SEQ_MATCHER:      # save the longest match 
        seq = SequenceMatcher(None, sl, l)
        i,j,k = seq.find_longest_match()
        if k >= CUTOFF_FOR_SQM*len(sl):
            #return np.array([x for x in range(j,j+k)])
            return [x for x in range(j,j+k)]
    else:
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
                #results.append((ind,ind+sll))
                #return np.array([x for x in range(ind,ind+sll)])
                return [x for x in range(ind,ind+sll)]
    return []

def vectorize_annotations(text, questions, answers, punct = True):
    """ Change the questions and answers to a list of indices in the original text.
      This function messes up \n annotations sometimes """
    
    # find indices
    if not punct:   #split by white space
        splitted_text = text.split()
        indices_questions = [find_sub_list(i.split(), splitted_text) for i in questions]
        indices_answers = [find_sub_list(i.split(), splitted_text) for i in answers]
    else:           # split by white space and punctuation
        splitted_text = re.findall(r'\w+|[^\s\w]+', text)
        indices_questions = [find_sub_list(re.findall(r'\w+|[^\s\w]+', i), splitted_text) for i in questions]
        indices_answers = [find_sub_list(re.findall(r'\w+|[^\s\w]+', i), splitted_text) for i in answers]

    vect_text = np.empty(len(splitted_text), dtype=int)
    vect_text[:] = label2id["O"]
    vect_text[flatten(indices_questions)] = label2id["Q"]
    vect_text[flatten(indices_answers)]= label2id["A"]

    # tolist() to make seqeval work
    return splitted_text, vect_text.tolist(), indices_questions, indices_answers


def read_annotated_file(path, key="text"):
    data = []
    
    with open(path, "r") as f:
        for line in f:
            d = json.loads(line)
            result = {"text":d["text_plain"], "q":[], "a": []}
            val = d[key]
            if not val:
                continue
            for el in val:
                if "q" in el:
                    result["q"].append(el["q"])
                elif "a" in el:
                    result["a"].append(el["a"])
            tokens, tags, indices_q, indices_a = vectorize_annotations(d["text_plain"], result["q"], result["a"])
            data.append({"id":d["id"], "text": d["text_plain"], "tokens":tokens, "tags":tags})
    
    return data


def remove_special_tokens_original(label, prediction):
    token_predictions = [
        [tag_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    token_labels = [
        [tag_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return token_labels, token_predictions

def remove_special_tokens(label, prediction):
    label_new = [i for i in label if i != -100]
    prediction_new = [prediction[i] for i in range(len(prediction)) if label[i] != -100]
    return label_new, prediction_new
    

#...............................................driver..........................................#

def main():
    options = argparser().parse_args(sys.argv[1:])
    log("Loading tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(options.tokenizer_path)
    log("...done") 
    log("Loading model")
    model_path = MODEL_PREFIX+options.model_path
    model = transformers.AutoModelForTokenClassification.from_pretrained(model_path)
    model.to(DEVICE)
    log("...done")
    
    
    if options.input_lang=="detect":
        # the data to use is in the model name after first "_"
        options.input_lang=options.model_path.split("_")[1]

    
    if options.input_lang=="en":
        options.input = DATA_EN
        #options.input_format = "dataset"
        #options.input_key="test"
        options.input_format = "jsonl"
        options.input_key = "_"
        
    elif options.input_lang=="fi":
        options.input = DATA_FI
        options.input_format= "jsonl"
        options.input_key = "-"
    
        
        
    options.output = "/scratch/project_2002026/amanda/register-qa/evaluation/final_3_results/predictions/"+options.model_path+"_special_"+str(options.remove_special_tokens)+"_preds_with_"+options.input_lang+".jsonl"

    
    # if the data is locally in a dataset format
    if options.input_format == "dataset":
        dataset = load_from_disk(options.input)
        outfile = open(options.output, 'w')
        for d in dataset[options.input_key]:
            text = " ".join(d["tokens"])
            true_labels = d["tags"]
            tokenized, tokenized_labels = tokenize_and_align_labels(d, tokenizer, options)#.to(DEVICE)
            tokenized = tokenized.to(DEVICE)
            pred_labels = predict_labels(tokenized, model, options)
            if options.remove_special_tokens:
                tokenized_labels, pred_labels = remove_special_tokens(tokenized_labels, pred_labels[0])
            outfile.write(str({"id": d["q_id"],
                               "text":text, 
                               "tokens": tokenized["input_ids"].tolist(), 
                               "labels": [id2label[i] for i in tokenized_labels[:MAX_LENGTH]], 
                               "preds": [id2label[i] for i in pred_labels]})+"\n")
                                                                                                    
        outfile.close()
        return 
            
        
    # if the data is in the annotated jsonl format
    if options.input_format == "jsonl":
        dataset = read_annotated_file(options.input)
        outfile = open(options.output, 'w')
        for d in dataset:
            text = d["text"]
            true_labels = d["tags"]
            tokenized, tokenized_labels = tokenize_and_align_labels(d, tokenizer, options)#.to(DEVICE)
            tokenized = tokenized.to(DEVICE)
            pred_labels = predict_labels(tokenized, model, options)
            if options.remove_special_tokens:
                tokenized_labels, pred_labels = remove_special_tokens(tokenized_labels, pred_labels[0])
            outfile.write(str({"id": d["id"],
                               "text":text, 
                               "tokens": tokenized["input_ids"].tolist(), 
                               "labels": [id2label[i] for i in tokenized_labels[:MAX_LENGTH]], 
                               "preds": [id2label[i] for i in pred_labels]})+"\n")

        outfile.close()
        return 
    

if __name__ == "__main__":
    main()