import comet_ml
import transformers
import datasets
import torch
import argparse
from pprint import PrettyPrinter
import logging
import numpy as np

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
        description="A script making a register identification classifier.",
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
    parser.add_argument('--batch', type=int, default=8,
        help="The batch size for the model. Defaults to 8.")
    parser.add_argument('--epochs', type=int, default=3,
        help="The number of epochs to train for. Defaults to 3.")
    parser.add_argument('--learning', type=float, default=8e-6,
        help="The learning rate for the model. Defaults to 8e-6.")
    parser.add_argument('--save', action="store_true", default=False,
        help="Whether to save the model.")
    parser.add_argument('--save_name', type=str, default="register-qa-binary",
        help="The name for the model to save.")
    parser.add_argument('--weights', action="store_true", default=False,
        help="Whether to use class weights or not for the loss function.")

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

print("qa in train",len(qa["train"]))
print("qa in dev",len(qa["dev"]))
print("qa in test",len(qa["test"]))


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


def make_class_weights(train):
    """Calculates class weights for the loss function based on the train split."""
    from sklearn.utils import class_weight
    #print(np.unique(np.ravel(train["label"], order='C')))
    #print(np.ravel(np.array(train["label"]), order='C')[:10])
    labels = train["label"]
    print(labels[:10])
    print(np.unique(labels))
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels) # only the first is a positional arguments blah
    print(class_weights)
    return class_weights


class_weights = make_class_weights(dataset["train"])
class_weights = torch.tensor(class_weights).to("cuda:0")

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    print(classification_report(labels, preds, target_names=unique_labels))
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

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
class NewTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if args.weights == True:
            loss_fct = torch.nn.CrossEntropyLoss(weight = class_weights.float())
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
            labels.view(-1))
        return (loss, outputs) if return_outputs else loss


num_labels = len(unique_labels)
print(num_labels)

model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir="../new_cache_dir/", id2label=id2label, label2id=label2id)

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
trainer = NewTrainer(
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
print(eval_results)
print('F1:', eval_results['eval_f1'])

# this part is now unnecessary as I print the classification report in the compute multilabel metrics method
# # see how the labels are predicted
# test_pred = trainer.predict(dataset['test'])
# trues = test_pred.label_ids
# predictions = test_pred.predictions

# if args.threshold == None:
#     threshold = optimize_threshold(predictions, trues)
# else:
#     threshold = args.threshold
# sigmoid = torch.nn.Sigmoid()
# probs = sigmoid(torch.Tensor(predictions))
# # next, use threshold to turn them into integer predictions
# preds = np.zeros(probs.shape)
# preds[np.where(probs >= threshold)] = 1

# print(classification_report(trues, preds, target_names=unique_labels))

if args.save == True:
    trainer.model.save_pretrained(f"models/{args.save_name}")