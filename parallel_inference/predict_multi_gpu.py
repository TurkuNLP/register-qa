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

# This was originally made by Risto Luukkonen, edited by me

torch.set_num_threads(2)
MODEL_PATH = "finbert-base-fin-0.00002-MTv2.pt" # TODO change this, I use base model too now so that's good, inference does not take super long
TOKENIZER_PATH = "TurkuNLP/bert-base-finnish-cased-v1"

DEVICE = "cuda"
# TODO change this here to the labels I am now using, maybe add upper list too
LABELS_FULL = ['HI', 'ID', 'IN', 'IP', 'LY', 'MT', 'NA', 'OP', 'SP', 'av', 'ds', 'dtp', 'ed', 'en', 'fi', 'it', 'lt', 'nb', 'ne', 'ob', 'ra', 're', 'rs', 'rv', 'sr']
DEVICE_COUNT = torch.cuda.device_count()

def log(s):
    print(s, flush=True)

def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--tokenizer_path', default=TOKENIZER_PATH,
                    help='Pretrained model name')
    ap.add_argument('--batch_size_per_device', default=32, type=int, help="batch_size per gpu")
    ap.add_argument('--text', metavar='FILE', required=True,
                    help='Text to be predicted') #could also be string?
    ap.add_argument('--file_type', choices=['tsv', 'jsonl', 'txt'], default='jsonl')
    ap.add_argument('--model_path', default=MODEL_PATH, metavar='FILE',
                    help='Load model from file')
    # TODO check threshold, for training currently using 0.5 instead of optimization actually
    ap.add_argument('--threshold', default=0.45, metavar='FLOAT', type=float,
                    help='threshold for calculating f-score')
    ap.add_argument('--labels', choices=['full', 'upper'], default='full')
    ap.add_argument('--output', default="output_predictions.tsv", metavar='FILE', help='Location to save predictions')
    ap.add_argument('--id_col_name', default=None, type=str, help="which column to use as a id. For cc-fi it's \"id\"")
    return ap




def tokenize(batch, tokenizer):
    tokenized = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
    return tokenized

def predict_labels(data, model, options):
    with torch.no_grad():
        pred = model(**data)

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(pred.logits.cpu().detach().numpy()))
#    log("XXX probs", probs)
    # log(probs.shape)

    preds = np.zeros(probs.shape)
    preds[np.where(probs >= options.threshold)] = 1
    labellist = [ [LABELS_FULL[i] for i, val in enumerate(line) if val == 1] for line in preds]
    labellist = [ labels if len(labels) > 0 else ["No labels"] for labels in labellist]

    return labellist, probs.numpy()

# predict labels text at a time
#outf = open(options.text+'_preds.txt', 'w')

def load_text_batch(handle, options):
    ids = []
    batch = [handle.readline() for _ in range(0, options.batch_size)]
    batch = [line for line in batch if len(line) > 0]
    
    if options.file_type == "txt":
        texts = [line.split('\t')[1] for line in batch]
        assert "TODO: probably requires some work"

    elif options.file_type == "jsonl":
        texts = [json.loads(line)["text"] for line in batch]
        if options.id_col_name:
            ids = [json.loads(line)[options.id_col_name] for line in batch]
    
    else:
        assert "TODO"

    return ids, texts

def main():

    options = argparser().parse_args(sys.argv[1:])
    options.batch_size = options.batch_size_per_device * DEVICE_COUNT
    if os.path.exists(options.output):
        assert False, "Output-file exists. Overwriting not allowed. Please give a different outname or delete the old file."
    log(options)
    if options.labels == 'full':
        labels = LABELS_FULL 
    else:
        assert "TODO"
        # labels = labels_upper

    num_labels = len(labels)
    log(f"Number of labels: {num_labels}")

    # tokenizer from the pretrained model
    log("Loading tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(options.tokenizer_path)
    log("...done") 
    # load our fine-tuned model
    log("Loading model")
    model = torch.load(options.model_path, map_location=torch.device(DEVICE)) # TODO change this to from pretrained because I do not use pytorch to save my models, 
    model = DataParallel(model) # TODO  here the documentation says to use distributed parallelism but this works too https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead 
    # https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
    log("..done")
    log(f"Starting processing datastream from file: {options.text}")
    with open(options.text, 'r') as f:
        p_bar = tqdm(desc="Process batch: ")
        with open(options.output, 'w') as outfile:
            if options.id_col_name:
                header = [options.id_col_name, "labels", "text", *LABELS_FULL]
            else:
                header = ["labels", "text", *LABELS_FULL]
            
            writer = csv.writer(outfile, delimiter="\t")
            writer.writerow(header)
            
            ids, text_batch = load_text_batch(f, options)

            while text_batch:

                tokenized_batch = tokenize(text_batch, tokenizer).to(DEVICE)
                labels, scores = predict_labels(tokenized_batch, model, options) 

                for i in range(len(text_batch)):
                    
                    text = text_batch[i].replace("\n", "\\n")
                    if options.id_col_name:
                        line = [ ids[i],  " ".join(labels[i]), text ]
                    else:
                        line = [ " ".join(labels[i]), text ]
                    pred_list = [str(val) for val in scores[i]]
                    line = [*line, *pred_list]

                    writer.writerow(line)

                p_bar.update(len(text_batch))
                p_bar.refresh()
                
                ids, text_batch = load_text_batch(f, options)

    log("Finished processing file")
         

if __name__ == "__main__":
    main()
