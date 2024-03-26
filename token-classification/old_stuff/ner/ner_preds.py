import transformers
import datasets
from transformers.pipelines.pt_utils import KeyDataset
from pprint import PrettyPrinter
from tqdm.auto import tqdm
import gzip
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

pprint = PrettyPrinter(compact=True).pprint


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model',
                    help='Pretrained model name')
    ap.add_argument('--data', metavar='FILE', required=True,
                    help='data file to predict on')
    ap.add_argument('--save',type=str, required=True,
                    help='name of the file to save')
    return ap

args = argparser().parse_args()

def load_dataset(name, files):
    dset = datasets.load_dataset(
        "csv",
        data_files={name: files},
        delimiter="\t",
        split=name, # so that it returns a dataset instead of datasetdict
        column_names=['id', 'label', 'text', 'prob1', 'prob2'],
        features=datasets.Features({    # Here we tell how to interpret the attributes
        "text":datasets.Value("string"),
        "label":datasets.ClassLabel(names=["QA_NEW", "NOT_QA"]),
        "id":datasets.Value("string"),
        "prob1":datasets.Value("float"),
        "prob2":datasets.Value("float"),
         }))
    return dset

dataset = load_dataset("train", args.data)

def replace_n(example):
  example["text"] = example["text"].replace("\n", " ")
  example["text"] = example["text"].replace("\\n", " ")

  return example

dataset = dataset.map(replace_n)

 
#stride=50, 
token_classifier = transformers.pipeline("ner", model=args.model, tokenizer="xlm-roberta-base",aggregation_strategy="average", ignore_labels=[""],device=0)
#aggregation_strategy="simple"

tokens = token_classifier(KeyDataset(dataset, "text"))

texts = dataset["text"]
ids = dataset["id"]

all_found = []
for idx, (extracted_entities, text, idd) in tqdm(enumerate(zip(tokens, texts, ids)), total=len(tokens)):
    one_found = []
    for entity in extracted_entities:
        if entity["entity_group"] == "QUESTION":
            temp = dataset[idx]["text"][entity["start"]:entity["end"]] #entity["word"]
            one_found.append({"q": temp})
        elif entity["entity_group"] == "ANSWER":
            temp = dataset[idx]["text"][entity["start"]:entity["end"]] #entity["word"]
            one_found.append({"a":temp})
        elif entity["entity_group"] == "O":
            temp = dataset[idx]["text"][entity["start"]:entity["end"]] #entity["word"]
            one_found.append({"t":temp})

    all_found.append(one_found)


multiple=open(args.save,"wt")
for idd, text, one in zip(ids, texts, all_found):
    #print(text_id)
    save = {}
    save["id"] = idd
    save["text"] = one
    save["text_plain"] = text
    line=json.dumps(save,ensure_ascii=False)
    print(line,file=multiple)