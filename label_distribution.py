import json
import datasets
import argparse
from pprint import PrettyPrinter
import numpy as np

pprint = PrettyPrinter(compact=True).pprint

# python3 label_distribution.py --train_set data/CORE-corpus/train.tsv.gz --dev_set data/CORE-corpus/dev.tsv.gz --test_set data/CORE-corpus/test.tsv.gz

def arguments():
    #parser for the optional arguments related to hyperparameters
    parser = argparse.ArgumentParser(
        description="A script making a register identification classifier.",
        epilog="Made by Anni Eskelinen"
    )
    parser.add_argument('--train_set', nargs="+", required=True,
        help="Give at least one or more train files separated by a space.")
    parser.add_argument('--dev_set', nargs="+", required=True,
        help="Give at least one or more dev files separated by a space.")
    parser.add_argument('--test_set', nargs="+", required=True,
        help="Give at least one or more test files separated by a space.")

    args = parser.parse_args()

    return args 

args = arguments()
pprint(args)


# the data is fitted to these labels
unique_labels = ["NOT_QA", "QA_NEW"] 



def load_dataset(name, files):
    # it is possible to load zipped csv files like this according to documentation: https://huggingface.co/docs/datasets/loading#csv
    dset = datasets.load_dataset(
        "csv", 
        data_files={name: files},
        delimiter="\t",
        split=name, # so that it returns a dataset instead of datasetdict
        column_names=['label', 'text'],
        features=datasets.Features({    # Here we tell how to interpret the attributes
        "text":datasets.Value("string"),
        "label":datasets.Value("string")})
        )

    # remember to shuffle because the data is in en,fi,fre,swe order! (or whatever language order)
    dset.shuffle(seed=1234)

    return dset

train = load_dataset("train", args.train_set)
dev = load_dataset("dev", args.dev_set)
test = load_dataset("test", args.test_set)
 

dataset = datasets.DatasetDict({"train":train, "dev":dev, "test":test})
shuffled_dataset = dataset.shuffle(seed=42)
print(dataset)
#print(dataset["train"][0])


# HERE CHANGE THE MAPPING TO MAKE MULTICLASS (BINARY)

#register scheme mapping:
sub_register_map = {
    'NA': 'NOT_QA',
    'NE': 'NOT_QA',
    'SR': 'NOT_QA',
    'PB': 'NOT_QA',
    'HA': 'NOT_QA',
    'FC': 'NOT_QA',
    'TB': 'NOT_QA',
    'CB': 'NOT_QA',
    'OA': 'NOT_QA',
    'OP': 'NOT_QA',
    'OB': 'NOT_QA',
    'RV': 'NOT_QA',
    'RS': 'NOT_QA',
    'AV': 'NOT_QA',
    'IN': 'NOT_QA',
    'JD': 'NOT_QA',
    'FA': 'QA_NEW', #fi
    'DT': 'NOT_QA',
    'IB': 'NOT_QA',
    'DP': 'NOT_QA',
    'RA': 'NOT_QA',
    'LT': 'NOT_QA',
    'CM': 'NOT_QA',
    'EN': 'NOT_QA',
    'RP': 'NOT_QA',
    'ID': 'NOT_QA',
    'DF': 'NOT_QA',
    'QA': 'QA_NEW', # ID
    'HI': 'NOT_QA',
    'RE': 'NOT_QA',
    'IP': 'NOT_QA',
    'DS': 'NOT_QA',
    'EB': 'NOT_QA',
    'ED': 'NOT_QA',
    'LY': 'NOT_QA',
    'PO': 'NOT_QA',
    'SO': 'NOT_QA',
    'SP': 'NOT_QA',
    'IT': 'NOT_QA',
    'FS': 'NOT_QA',
    'TV': 'NOT_QA',
    'OS': 'NOT_QA',
    'IG': 'NOT_QA',
    'MT': 'MT', # keep this as mt to take them out later
    'HT': 'NOT_QA',
    'FI': 'QA_NEW', # fi
    'OI': 'NOT_QA',
    'TR': 'NOT_QA',
    'AD': 'NOT_QA',
    'LE': 'NOT_QA',
    'OO': 'NOT_QA',
    'MA': 'NOT_QA',
    'ON': 'NOT_QA',
    'SS': 'NOT_QA',
    'OE': 'NOT_QA',
    'PA': 'NOT_QA',
    'OF': 'NOT_QA',
    'RR': 'NOT_QA',
    'FH': 'QA_NEW', # HI
    'OH': 'NOT_QA',
    'TS': 'NOT_QA',
    'OL': 'NOT_QA',
    'PR': 'NOT_QA',
    'SL': 'NOT_QA',
    'TA': 'NOT_QA',
    'OTHER': 'NOT_QA'
}


def split_labels(dataset):
    # NA ends up as None because NA means that there is nothing (not available)
    # so we have to fix it
    if dataset['label'] == None:
        dataset['label'] = np.array('NOT_QA') # NA
    else:
        dataset['label'] = np.array(dataset['label'].split())
        mapped = [sub_register_map[l] if l not in unique_labels else l for l in dataset['label']] # added for full
        dataset['label'] = np.array(sorted(list(set(mapped)))) # added for full
    return dataset

print("sub-register mapping")
dataset = dataset.map(split_labels)

print("filtering")
# remove comments that have MT label, all, not just the ones with QA_NEW (could change to keep this but take out only ones with QA_NEW)
filtered_dataset = dataset.filter(lambda example: "MT" not in example['label']) 
#filtered_dataset = filtered_dataset.filter(lambda example: 'QA_NEW' not in example['label']) #try to take out the rest of multilabel things -> actually why would it be necessary :D
#filtered_dataset = filtered_dataset.filter(lambda example: "OS" not in example['label']) # now no warnings for this one either
filtered_dataset = filtered_dataset.filter(lambda example: example["text"] != None) # filter empty text lines from english train (and others?)
# would be more efficient to filter before I ever use in this code but ehh

print("total train", len(filtered_dataset["train"]))
print("total test", len(filtered_dataset["test"]))
print("total dev", len(filtered_dataset["dev"]))

# do this so that the model has these ready for easy pipeline usage
id2label = dict(zip(range(len(unique_labels)), unique_labels))
label2id = dict(zip(unique_labels, range(len(unique_labels))))

print("change labels to only qa")
# change register to just qa
# this maps from the string label to int 0 or 1
def only_qa(dataset):
    if "NOT_QA" and "QA_NEW" in dataset["label"]:
        dataset["label"] = label2id["QA_NEW"]
    elif "NOT_QA" in dataset["label"]:
        dataset["label"] = label2id["NOT_QA"]
    elif "QA_NEW" in dataset["label"]:
        dataset["label"] = label2id["QA_NEW"]
    # this if there is NA so not found label because of mapping weirdness or something
    else:
        dataset["label"] = label2id["NOT_QA"]

    return dataset

dataset = filtered_dataset.map(only_qa)

print(dataset["train"]["label"][:10])

# check how many qa in train data
qa = dataset.filter(lambda example: example['label'] == 1)
not_qa = dataset.filter(lambda example: example['label'] == 0)

print("qa in train",len(qa["train"]))
print("qa in dev",len(qa["dev"]))
print("qa in test",len(qa["test"]))

print("not qa in train",len(not_qa["train"]))
print("not qa in dev",len(not_qa["dev"]))
print("not qa in test",len(not_qa["test"]))
