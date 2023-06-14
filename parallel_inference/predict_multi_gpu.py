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
MODEL_PATH = "./../models/new_model2" #"models/new_model" #I use base model too now so that's good, inference does not take super long
 #"finbert-base-fin-0.00002-MTv2.pt"
TOKENIZER_PATH = "xlm-roberta-base" #"TurkuNLP/bert-base-finnish-cased-v1"

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
    ap.add_argument('--id_col_name', default=None, type=str, help="which column to use as a id. For cc-fi and parsebank it's \"id\"")
    ap.add_argument('--long_text', default=False, action="store_true",
                    help='whether to split long texts and predict labels for each part.')
    return ap



def tokenize(batch, tokenizer):
    tokenized = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
    return tokenized

def tokenize_split(batch, tokenizer):
    def tokenize(example):
        return tokenizer(
            example,
            return_tensors="pt",
            add_special_tokens=False,
            # add also offset mapping stuff
            return_offsets_mapping=True, # give where the tokens start and end
            #return_special_tokens_mask=False #tell so that there are no seps etc.
        )
    # this tokenizes the whole thing
    tokens =list(map(tokenize, batch))



    chunksize = 512

    # for loop, go through each text to split
    all_inputs = []
    for token in tokens:
        # split into chunks of 510 tokens, we also convert to list (default is tuple which is immutable)
        input_id_chunks = list(token['input_ids'][0].split(chunksize - 2))
        mask_chunks = list(token['attention_mask'][0].split(chunksize - 2))
        offset_chunks = list(token['offset_mapping'][0].split(chunksize - 2))

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
                # # have to pad because stack expects same dimensions (length is the problem)
                # offset_chunks[i] = torch.cat([
                #     offset_chunks[i], torch.Tensor([0] * pad_len+2)
                # ])

        input_ids = torch.stack(input_id_chunks)
        attention_mask = torch.stack(mask_chunks)
        #offset_mappings = torch.stack(offset_chunks) # no need to do anything else since I just use this to index chars
        offset_mappings = torch.cat(offset_chunks, dim=0)

        input_dict = {
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask.int(),
            'offset_mapping': offset_mappings.int()
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
    probs = torch.empty(0,9) # does this work or not?
    remove = []

    # get shorter texts for the different.tsv file
    # TODO figure out why there is a problem with the index below where I put another TODO
    # shortened = []
    # for i in range(len(text_batch)):
    #     begin = 0
    #     temp = []
    #     for j in range(len(data[i])):
    #         num = len(data[j]["offset_mapping"])-1 # have to find where there is 0's so I can take the offsets before that
    #         temp.append(text_batch[i][begin:num])
    #         begin = num # to continue from
    #     shortened.append(temp)
        
    for i in range(len(data)): # or double loop if this does not work
        data[i].pop("offset_mapping")

    with torch.no_grad():
        # here needs for loop because will be list of lists instead of list
        for i, text in enumerate(data):
            different = False
            pred = model(**text)

            sigmoid = torch.nn.Sigmoid()
            new_probs = sigmoid(torch.Tensor(pred.logits.cpu().detach().numpy()))
            # check the labels
            preds = np.zeros(new_probs.shape)
            preds[np.where(new_probs >= options.threshold)] = 1

            # if labels are different for each part of the text
            labels = [ [labels_used[i] for i, val in enumerate(line) if val == 1] for line in preds] 
            labels = [ labell if len(labell) > 0 else ["No labels"] for labell in labels]
            # compare the predictions for the different parts of the texts
            for j in range(len(preds)):
                if len(preds) == 1:
                    break
                if j == 1:
                    continue
                # if they are different save them to a different file (same way basically as in the main() method
                if not np.array_equal(preds[j], preds[j-1]): 
                    different = True
                    #print("different")
                    probs2 = new_probs.numpy() # this is just if different and need to get the probs for file

                    # set the texts and predictions and labels like in the main one :D
                    #text = shortened[i][j].replace("\n", "\\n") #TODO index out of range? HOW CAN THIS BE WHEN THIS AND DATA DIMENSIONS ARE THE SAMEEE
                    text = text_batch[i][j].replace("\n", "\\n")
                    if options.id_col_name:
                        line = [ ids[i],  " ".join(labels[j]), text ]
                        line2 = [ ids[i],  " ".join(labels[j-1]), text ]
                    else:
                        line = [ " ".join(labels[j]), text ] 
                        line2 = [ " ".join(labels[j-1]), text ] 
                    pred_list = [str(val) for val in probs2[j]]
                    line = [*line, *pred_list]

                    with open("different.tsv", 'a') as outfile: # change from w to a so no overwriting
                        different_writer = csv.writer(outfile, delimiter="\t")
                        different_writer.writerow(line)
                        different_writer.writerow(line2)

            if different == True:
                remove.append(i)
            # just take one part if the labels are the same
            # and skip if there was many label combinations for one text
            if different == False:
                prob = new_probs[0]
                prob = torch.unsqueeze(prob,0) # make this two dimensional

                probs = torch.cat((probs, prob), 0)

    # remove the texts that have removed probabilities
    for index in sorted(remove, reverse=True):
        text_batch.pop(index)
        ids.pop(index)

    preds = np.zeros(probs.shape)
    preds[np.where(probs >= options.threshold)] = 1
    labellist = [ [labels_used[i] for i, val in enumerate(line) if val == 1] for line in preds] 
    labellist = [ labels if len(labels) > 0 else ["No labels"] for labels in labellist]

    return labellist, ids, text_batch, probs.numpy()

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
                # maybe I can just make the batch size smaller, but not sure how much there is these long texts
                # that's one thing I could check from the data before I do this but hmmm

                if options.long_text:
                    tokenized_batch = tokenize_split(text_batch, tokenizer)
                    # # tokenized batch is now a list so cannot move to device, have to loop and put them all into device
                    for input in tokenized_batch:
                        for key, value in input.items():
                            input[key] = input[key].to(DEVICE)
                    labels, ids, text_batch, scores = predict_split_labels(tokenized_batch, model, options,text_batch, ids, labels_used) 

                else:
                    tokenized_batch = tokenize(text_batch, tokenizer).to(DEVICE)
                    labels, scores = predict_labels(tokenized_batch, model, options) 

                # TODO here is now list out of range as I do not put the different ones here oh noo
                # I have to delete those from the text_batch
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
