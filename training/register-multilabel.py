import comet_ml
import transformers
import datasets
import torch
import argparse
from pprint import PrettyPrinter
import logging
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


# disable dataset caching (only works on mapping and stuff, not for raw dataset -> have to force redownload)
#datasets.disable_caching()
datasets.enable_caching()

# get comet-ml variables from .env file
from dotenv import load_dotenv
load_dotenv()
# no need to use them otherwise so nothing else here


pprint = PrettyPrinter(compact=True).pprint
logging.disable(logging.INFO)

def arguments():
    #parser for the optional arguments related to hyperparameters
    parser = argparse.ArgumentParser(
        description="A script for getting register labeling benchmarks",
        epilog="Made by Anni Eskelinen"
    )
    parser.add_argument('--train_set', nargs="+", required=True,
        help="Give at least one or more train files separated by a space.")
    parser.add_argument('--dev_set', nargs="+", required=True,
        help="Give at least one or more dev files separated by a space.")
    parser.add_argument('--test_set', nargs="+", required=True,
        help="Give at least one or more test files separated by a space.")
    parser.add_argument('--model', default="xlm-roberta-large",
        help="Decide which model to use for training. Defaults to xlmr-large.")
    parser.add_argument('--threshold', type=float, default=None,
        help="The treshold which to use for predictions, used in evaluation. Defaults to 0.5.")
    parser.add_argument('--batch', type=int, default=8,
        help="The batch size for the model. Defaults to 8.")
    parser.add_argument('--epochs', type=int, default=3,
        help="The number of epochs to train for. Defaults to 3.")
    parser.add_argument('--learning', type=float, default=8e-6,
        help="The learning rate for the model. Defaults to 8e-6.")
    parser.add_argument('--save', action="store_true", default=False,
        help="Whether to save the model.")
    parser.add_argument('--weights', action="store_true", default=False,
        help="Whether to use class weights or not for the loss function.")

    args = parser.parse_args()

    return args 

args = arguments()
pprint(args)

# the data is fitted to these labels
#unique_labels = ["IN", "NA", "HI", "LY", "IP", "SP", "ID", "OP", QA_NEW""]
unique_labels = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP', 'av', 'ds', 'dtp', 'ed', 'en', 'it', 'lt', 'nb', 'ne', 'ob', 'ra', 're', 'rs', 'rv', 'sr', 'QA_NEW'] # 'fi', there should be nothing fi but just in case



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


#register scheme mapping:
sub_register_map = {
    'NA': 'NA',
    'NE': 'ne',
    'SR': 'sr',
    'PB': 'nb',
    'HA': 'NA',
    'FC': 'NA',
    'TB': 'nb',
    'CB': 'nb',
    'OA': 'NA',
    'OP': 'OP',
    'OB': 'ob',
    'RV': 'rv',
    'RS': 'rs',
    'AV': 'av',
    'IN': 'IN',
    'JD': 'IN',
    'FA': 'QA_NEW', #fi
    'DT': 'dtp',
    'IB': 'IN',
    'DP': 'dtp',
    'RA': 'ra',
    'LT': 'lt',
    'CM': 'IN',
    'EN': 'en',
    'RP': 'IN',
    'ID': 'ID',
    'DF': 'ID',
    'QA': 'QA_NEW', # ID
    'HI': 'HI',
    'RE': 're',
    'IP': 'IP',
    'DS': 'ds',
    'EB': 'ed',
    'ED': 'ed',
    'LY': 'LY',
    'PO': 'LY',
    'SO': 'LY',
    'SP': 'SP',
    'IT': 'it',
    'FS': 'SP',
    'TV': 'SP',
    'OS': 'OS',
    'IG': 'IP',
    'MT': 'MT',
    'HT': 'HI',
    'FI': 'QA_NEW', # fi
    'OI': 'IN',
    'TR': 'IN',
    'AD': 'OP',
    'LE': 'OP',
    'OO': 'OP',
    'MA': 'NA',
    'ON': 'NA',
    'SS': 'NA',
    'OE': 'IP',
    'PA': 'IP',
    'OF': 'ID',
    'RR': 'ID',
    'FH': 'QA_NEW', # HI
    'OH': 'HI',
    'TS': 'HI',
    'OL': 'LY',
    'PR': 'LY',
    'SL': 'LY',
    'TA': 'SP',
    'OTHER': 'OS'
}


def split_labels(dataset):
    # NA ends up as None because NA means that there is nothing (not available)
    # so we have to fix it
    if dataset['label'] == None:
        dataset['label'] = np.array('NA')
    else:
        dataset['label'] = np.array(dataset['label'].split())
        mapped = [sub_register_map[l] if l not in unique_labels else l for l in dataset['label']] # added for full
        dataset['label'] = np.array(sorted(list(set(mapped)))) # added for full
    return dataset

def binarize(dataset):
    mlb = MultiLabelBinarizer()
    mlb.fit([unique_labels]) # REMEMBER THIS FITS TO THE UNIQUE LABELS SO IF SOMETHING ISN'T THERE IT WILL BE DISCARDED
    dataset = dataset.map(lambda line: {'label': mlb.transform([line['label']])[0]})
    return dataset

print("sub-register mapping")
#pprint(dataset['train']['label'][:5])
dataset = dataset.map(split_labels)
#pprint(dataset['train']['label'][:5])

print("filtering")
# remove comments that have MT label
filtered_dataset = dataset.filter(lambda example: "MT" not in example['label']) 
filtered_dataset = filtered_dataset.filter(lambda example: "OS" not in example['label']) # now no warnings for this one either
filtered_dataset = filtered_dataset.filter(lambda example: example["text"] != None) # filter empty text lines from english train (and others?)
# would be more efficient to filter before I ever use in this code but ehh

print("train", len(filtered_dataset["train"]))
print("test", len(filtered_dataset["test"]))
print("dev", len(filtered_dataset["dev"]))

print("binarization")
dataset = binarize(filtered_dataset)
#pprint(dataset['train']['label'][:5])
#pprint(dataset['train'][:2])

# then use the tokenizer
model_name = args.model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        max_length=512,
        truncation=True # use some other method for this?
    )

print("tokenization")
dataset = dataset.map(tokenize)

num_labels = len(unique_labels)
print(num_labels)


# in case a threshold was not given, choose the one that works best with the evaluated data
def optimize_threshold(predictions, labels):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    best_f1 = 0
    best_f1_threshold = 0.5 # use 0.5 as a default threshold
    y_true = labels
    for th in np.arange(0.3, 0.7, 0.05):
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= th)] = 1
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = th
    return best_f1_threshold 

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels) # why is the sigmoid applies? could do without it
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    #next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels

    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    pre = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    rec = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)

    experiment = comet_ml.config.get_global_experiment()
    experiment.log_confusion_matrix(y_pred, y_true)

    # return as dictionary
    metrics = {'f1': f1_micro_average,
                'precision': pre,
                'recall': rec,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    if args.threshold == None:
        best_f1_th = optimize_threshold(preds, p.label_ids)
        threshold = best_f1_th
        print("Best threshold:", threshold)
    else:
        threshold=args.treshold
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids,
        threshold=threshold)
    return result

from collections import defaultdict

class LogSavingCallback(transformers.TrainerCallback):
    def on_train_begin(self, *args, **kwargs):
        self.logs = defaultdict(list)
        self.training = True

    def on_train_end(self, *args, **kwargs):
        self.training = False

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if self.training:
            for k, v in logs.items():
                if k != "epoch" or v not in self.logs[k]:
                    self.logs[k].append(v)

training_logs = LogSavingCallback()

early_stopping = transformers.EarlyStoppingCallback(
    early_stopping_patience=5
)

# we do this because for some reason the problem type does not change the loss function in the trainer
class MultilabelTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if args.weights == True:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight = class_weights)
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
            labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss



model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="multi_label_classification", cache_dir="../new_cache_dir/")

trainer_args = transformers.TrainingArguments(
    output_dir="checkpoints",
    evaluation_strategy="steps",
    logging_strategy="steps",  
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    num_train_epochs=args.epochs, # number of epochs = how many times the model has seen the whole training data
    learning_rate=args.learning,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=32,
)

data_collator = transformers.DataCollatorWithPadding(tokenizer)


# and finally train the model
trainer = MultilabelTrainer(
    model=model,
    args=trainer_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer = tokenizer,
    callbacks=[early_stopping, training_logs]
)

print("training")
trainer.train()


eval_results = trainer.evaluate(dataset["test"])
print('F1:', eval_results['eval_f1'])

# see how the labels are predicted
test_pred = trainer.predict(dataset['test'])
trues = test_pred.label_ids
predictions = test_pred.predictions

if args.threshold == None:
    threshold = optimize_threshold(predictions, trues)
else:
    threshold = args.threshold
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(torch.Tensor(predictions))
# next, use threshold to turn them into integer predictions
preds = np.zeros(probs.shape)
preds[np.where(probs >= treshold)] = 1

from sklearn.metrics import classification_report
print(classification_report(trues, preds, target_names=unique_labels))

if args.save == True:
    trainer.model.save_pretrained("../models/new_model")
    print("saved")