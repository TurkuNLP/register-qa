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

# This script was originally made by Risto Luukkonen, edited by me

torch.set_num_threads(2)
MODEL_PATH = "" #"models/new_model" #I use base model too now so that's good, inference does not take super long
TOKENIZER_PATH = "TurkuNLP/bert-base-finnish-cased-v1"

DEVICE = "cuda"
LABELS_UPPER = ["IN", "NA", "HI", "LY", "IP", "SP", "ID", "OP", "QA_NEW"] #added upper labels (with qa label)
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
    ap.add_argument('--long_text', default=False, action="store_true",
                    help='whether to split long texts and predict labels for each part.')
    return ap



def tokenize(batch, tokenizer):
    tokenized = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
    return tokenized

def tokenize_split(batch, tokenizer):
    # in the example here is only one text but this should be fine, later just have to loop these
    tokens = tokenizer.encode_plus(batch, add_special_tokens=False, 
                               return_tensors='pt')

    chunksize = 512

    # for loop, go through each text to split
    all_inputs = []
    for token in tokens:
        # split into chunks of 510 tokens, we also convert to list (default is tuple which is immutable)
        input_id_chunks = list(token['input_ids'][0].split(chunksize - 2))
        mask_chunks = list(token['attention_mask'][0].split(chunksize - 2))

        # loop through each chunk
        for i in range(len(input_id_chunks)):
            # add CLS and SEP tokens to input IDs
            input_id_chunks[i] = torch.cat([
                torch.tensor([101]), input_id_chunks[i], torch.tensor([102])
            ])
            # add attention tokens to attention mask
            mask_chunks[i] = torch.cat([
                torch.tensor([1]), mask_chunks[i], torch.tensor([1])
            ])
            # get required padding length
            pad_len = chunksize - input_id_chunks[i].shape[0]
            # check if tensor length satisfies required chunk size
            if pad_len > 0:
                # if padding length is more than 0, we must add padding
                input_id_chunks[i] = torch.cat([
                    input_id_chunks[i], torch.Tensor([0] * pad_len)
                ])
                mask_chunks[i] = torch.cat([
                    mask_chunks[i], torch.Tensor([0] * pad_len)
                ])

        input_ids = torch.stack(input_id_chunks)
        attention_mask = torch.stack(mask_chunks)


        input_dict = {
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask.int()
        }
        all_inputs.append(input_dict)

    return all_inputs

def predict_labels(data, model, options):
    with torch.no_grad():
        pred = model(**data)

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(pred.logits.cpu().detach().numpy()))
#    log("XXX probs", probs)
    # log(probs.shape)

    preds = np.zeros(probs.shape)
    preds[np.where(probs >= options.threshold)] = 1
    labellist = [ [LABELS_UPPER[i] for i, val in enumerate(line) if val == 1] for line in preds] # change to UPPER FOR QA, FULL for others
    labellist = [ labels if len(labels) > 0 else ["No labels"] for labels in labellist]

    return labellist, probs.numpy()

# predict labels text at a time
#outf = open(options.text+'_preds.txt', 'w')

def predict_split_labels(data, model, options, text_batch, ids, labels_used):
    probs = torch.Tensor() # does this work or not?
    with torch.no_grad():
        # here needs for loop because will be list of lists instead of list
        for i, text in enumerate(data):
            pred = model(**text)

            sigmoid = torch.nn.Sigmoid()
            new_probs = sigmoid(torch.Tensor(pred.logits.cpu().detach().numpy()))
            # check the labels
            preds = np.zeros(new_probs.shape)
            preds[np.where(new_probs >= options.threshold)] = 1

            probs2 = new_probs.numpy()
            # if labels are different for each part of the text

            labels = [ [labels_used[i] for i, val in enumerate(line) if val == 1] for line in preds] 
            labels = [ labell if len(labell) > 0 else ["No labels"] for labell in labels]
            for index, predicts in enumerate(preds):
                if predicts != preds[index-1]:
                    different = True
                    # set the texts and predictions and labels like in the main one :D
                    if options.id_col_name:
                        line = [ ids[i],  " ".join(labels[i]), text ]
                    else:
                        line = [ " ".join(labels[i]), text ] 
                    pred_list = [str(val) for val in probs2[i]]
                    line = [*line, *pred_list]

                     with open("different.tsv", 'w') as outfile:
                        writer = csv.writer(outfile, delimiter="\t")
                        writer.writerow(line)

                                
            # just take one part if the labels are the same
            if different == False:
                prob = new_probs[0]

            probs = torch.cat(probs, prob, 0)

    preds = np.zeros(probs.shape)
    preds[np.where(probs >= options.threshold)] = 1
    labellist = [ [labels[i] for i, val in enumerate(line) if val == 1] for line in preds] 
    labellist = [ labels if len(labels) > 0 else ["No labels"] for labels in labellist]

    return labellist, probs.numpy()

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
        labels_used = LABELS_FULL 
    elif options.labels == 'upper':
        labels_used = LABELS_UPPER
    else:
        assert "TODO"
        # labels = labels_upper

    num_labels = len(labels_used)
    log(f"Number of labels: {num_labels}")

    # tokenizer from the pretrained model
    log("Loading tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(options.tokenizer_path)
    log("...done") 
    # load our fine-tuned model
    log("Loading model")
    # model = torch.load(options.model_path, map_location=torch.device(DEVICE))
    model = transformers.AutoModelForSequenceClassification.from_pretrained(options.model_path)
    model.to(DEVICE)
    model = DataParallel(model) # TODO  here the documentation says to use distributed parallelism but this works too https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead 
    # https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
    log("..done")
    log(f"Starting processing datastream from file: {options.text}")
    with open(options.text, 'r') as f:
        p_bar = tqdm(desc="Process batch: ")
        with open(options.output, 'w') as outfile:
            if options.id_col_name:
                header = [options.id_col_name, "labels", "text", *labels_used]
            else:
                header = ["labels", "text", *labels_used]
            
            writer = csv.writer(outfile, delimiter="\t")
            writer.writerow(header)
            
            ids, text_batch = load_text_batch(f, options)

            while text_batch:

                # MAYBEE HAVE TO CHANGE TEXT_LOAD ABOVE DUE TO THE BATCHING IT MAKES, WHAT I DO MIGHT RUIN IT
                # note that if changing batch, have to make sure that a text is not split to separate batches
                # maybe I can just make the batch size smaller, but not sure how much there is these long texts
                #(ask risto?)

                if options.long_text:
                    tokenized_batch = tokenize_split(text_batch, tokenizer).to(DEVICE)
                    labels, scores = predict_split_labels(tokenized_batch, model, options,text_batch, ids, labels_used) 

                else:
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
