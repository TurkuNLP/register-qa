import transformers
import datasets
import torch
import argparse
from pprint import PrettyPrinter
import logging
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

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
    parser.add_argument('--test_set', nargs="+", required=True,
        help="Give at least one or more test files separated by a space.")
    parser.add_argument('--full', action='store_true', default=False,
        help="Decide whether or not to use full labels or only upper labels. Defaults to upper labels.")
    parser.add_argument('--model', default="xlm-roberta-large",
        help="Decide which model to use for training. Defaults to xlmr-large.")
    parser.add_argument('--treshold', type=float, default=0.5,
        help="The treshold which to use for predictions, used in evaluation. Defaults to 0.5."
    )
    parser.add_argument('--batch', type=int, default=8,
        help="The batch size for the model. Defaults to 8."
    )
    parser.add_argument('--epochs', type=int, default=3,
        help="The number of epochs to train for. Defaults to 3."
    )
    parser.add_argument('--learning', type=float, default=8e-6,
        help="The learning rate for the model. Defaults to 8e-6."
    )
    parser.add_argument('--multilingual', action='store_true', default=False,
        help="Decide whether or not to save the model. Defaults to not saving.")
    parser.add_argument('--saved', default="saved_models/all_multilingual",
        help="Give a new path for the saved model.")
    parser.add_argument('--checkpoint', default="../multilabel/checkpoints",
        help="Give a new path for the checkpoints")
    parser.add_argument('--lang', default="",
        help="Give a name to the saved plots.")
    args = parser.parse_args()

    return args 

args = arguments()
pprint(args)

# the data is fitted to these labels
if(args.full == False):
    unique_labels = ["IN", "NA", "HI", "LY", "IP", "SP", "ID", "OP"]
else:
    unique_labels = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP', 'av', 'ds', 'dtp', 'ed', 'en', 'fi', 'it', 'lt', 'nb', 'ne', 'ob', 'ra', 're', 'rs', 'rv', 'sr']


# it is possible to load zipped csv files like this according to documentation: https://huggingface.co/docs/datasets/loading#csv
train = datasets.load_dataset(
    "csv", 
    data_files={'train':args.train_set},
    delimiter="\t",
    split='train', # so that it returns a dataset instead of datasetdict
    column_names=['label', 'text'],
    features=datasets.Features({    # Here we tell how to interpret the attributes
      "text":datasets.Value("string"),
      "label":datasets.Value("string")})
    )

# remember to shuffle because the data is in en,fi,fre,swe order!

train.shuffle(seed=1234)

# this also shuffles by default (should I set a seed? or not shuffle anymore?) shuffle=False or seed=1234
#train, dev = train.train_test_split(test_size=0.2).values()

test = datasets.load_dataset(
    "csv", 
    data_files={'test':args.test_set}, 
    delimiter="\t",
    split='test',
    column_names=['label', 'text'],
    features=datasets.Features({    # Here we tell how to interpret the attributes
      "text":datasets.Value("string"),
      "label":datasets.Value("string")})
    )

dataset = datasets.DatasetDict({"train":train, "test":test}) #"dev":dev,
pprint(dataset)


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
    'FA': 'fi',
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
    'QA': 'ID',
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
    'FI': 'fi',
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
    'FH': 'HI',
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
    mlb.fit([unique_labels])
    dataset = dataset.map(lambda line: {'label': mlb.transform([line['label']])[0]})
    return dataset

#pprint(dataset['train']['label'][:5])
dataset = dataset.map(split_labels)
#pprint(dataset['train']['label'][:5])
dataset = binarize(dataset)
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

dataset = dataset.map(tokenize)

num_labels = len(unique_labels)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="multi_label_classification", cache_dir="../new_cache_dir/")
# these are in eval mode already and documentation says to change it to train but is that really necessary? it has worked with eval on but I should try stuff
#model.train()
# model.eval() 

trainer_args = transformers.TrainingArguments(
    args.checkpoint, # change this to put the checkpoints somewhere else
    evaluation_strategy="epoch",
    logging_strategy="epoch",  # number of epochs = how many times the model has seen the whole training data
    save_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=args.epochs,
    learning_rate=args.learning,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=32,
)

data_collator = transformers.DataCollatorWithPadding(tokenizer)

#compute accuracy and loss
from transformers import EvalPrediction
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels):
    # the treshold has to be really low because the probabilities of the predictions are not great, could even do without any treshold then? or find one that works best between 0.1 and 0.5
    threshold=args.treshold

    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels) # why is the sigmoid applies? could do without it
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    #next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
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
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

# and finally train the model
trainer = MultilabelTrainer(
    model=model,
    args=trainer_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer = tokenizer,
    callbacks=[early_stopping, training_logs]
)

trainer.train()

# plot to see how the loss and evaluation go
import matplotlib.pyplot as plt

def plot(logs, keys, labels, filename):
    values = sum([logs[k] for k in keys], [])
    plt.ylim(max(min(values)-0.1, 0.0), min(max(values)+0.1, 1.0))
    for key, label in zip(keys, labels):    
        plt.plot(logs["epoch"], logs[key], label=label)
    plt.legend()
    plt.show()
    plt.savefig(filename) # set file name where to save the plots
    plt.close() # this closes the current figure so that no texts are left hanging in the next one

plot(training_logs.logs, ["loss", "eval_loss"], ["Training loss", "Evaluation loss"], "logs/"+ args.lang +"_loss.jpg")
plot(training_logs.logs, ["eval_f1"], ["Evaluation F1-score"], "logs/"+ args.lang +"_f1.jpg")





if args.multilingual == True:
    trainer.model.save_pretrained(args.saved)
else:
    model.eval()
    eval_results = trainer.evaluate(dataset["test"])
    print('F1:', eval_results['eval_f1'])

    # see how the labels are predicted
    test_pred = trainer.predict(dataset['test'])
    trues = test_pred.label_ids
    predictions = test_pred.predictions

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    preds = np.zeros(probs.shape)
    preds[np.where(probs >= args.treshold)] = 1

    from sklearn.metrics import classification_report
    print(classification_report(trues, preds, target_names=unique_labels))