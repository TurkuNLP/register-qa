from pprint import PrettyPrinter
import logging
import datasets
import transformers
import torch
import seqeval
import evaluate
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

print(torch.cuda.is_available()) # this does not find cuda aaaah how to fix, no wonder everything has been slow lol

logging.disable(logging.INFO)
pprint = PrettyPrinter(compact=True).pprint

def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name',
                    help='Pretrained model name')
    ap.add_argument('--train', metavar='FILE', required=True,
                    help='train file')
    ap.add_argument('--test', metavar='FILE', required=True,
                    help='test file')
    ap.add_argument('--dev', metavar='FILE', required=True,
                    help='dev file')
    ap.add_argument('--batch', type=int, default=8,
        help="The batch size for the model")
    ap.add_argument('--epochs', type=int, default=3,
        help="The number of epochs to train for")
    ap.add_argument('--lr', type=float, default=8e-6,
        help="The learning rate for the model")
    ap.add_argument('--save', type=str, default=None,
        help="If used the model will be saved with the string name given")
    return ap

args = argparser().parse_args()

#label_names = ["O", "QB", "QI", "AB", "AI"]

# dataset = datasets.load_dataset("json", 
#     data_files={'train': args.train, 'validation': args.dev, 'test': args.test},
#     features=datasets.Features({    # Here we tell how to interpret the attributes
#         "tags":datasets.Sequence(datasets.ClassLabel(num_classes=5, names=label_names)),
#         "tokens":datasets.Sequence(datasets.Value("string")),
#         "q_id":datasets.Value("string"),
#         "subreddit":datasets.Value("string")
#     }))

dataset = datasets.load_from_disk("../data/qa_token_classification/dataset_answer_new")

dataset=dataset.shuffle()

print(dataset)
#pprint(dataset["train"][0])


tag_names = dataset["train"].features["tags"].feature.names
pprint(tag_names)
print(len(tag_names))

print("tokenization")
tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)

def tokenize_and_align_labels(example):
    # adapted from https://huggingface.co/docs/transformers/custom_datasets#tok_ner
    tokenized = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True)
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
            labels.append(-100)
        previous_word_idx = word_idx

    tokenized["labels"] = labels
    return tokenized

# Apply the tokenizer to the whole dataset using .map()
dataset = dataset.map(tokenize_and_align_labels)

#pprint(dataset["train"][0])

# make id2label automatically
id2label = {}
for id, label in zip(list(range(len(tag_names))), tag_names):
    id2label[id] = label
label2id = {}
for label, id in zip(tag_names, list(range(len(tag_names)))):
    label2id[label] = id


model = transformers.AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(tag_names), id2label=id2label, label2id=label2id)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

trainer_args = transformers.TrainingArguments(
    "checkpoints/answer/",
    evaluation_strategy="steps",
    logging_strategy="steps",
    load_best_model_at_end=True,
    eval_steps=2500,
    logging_steps=2500,
    save_steps=2500,
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=16,
    num_train_epochs=args.epochs,
)

metric = evaluate.load("seqeval")
#change from accuracy to -> seqeval

from seqeval.scheme import IOB2
def compute_metrics(outputs_and_labels):
    outputs, labels = outputs_and_labels
    predictions = outputs.argmax(axis=2)

    # Remove ignored indices (special tokens)
    token_predictions = [
        [tag_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    token_labels = [
        [tag_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = metric.compute(
        predictions=token_predictions,
        references=token_labels,
        #suffix=True, scheme="IOB2"
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }


data_collator = transformers.DataCollatorForTokenClassification(tokenizer)
# Argument gives the number of steps of patience before early stopping
early_stopping = transformers.EarlyStoppingCallback(
    early_stopping_patience=5
)

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



trainer = transformers.Trainer(
    model=model,
    args=trainer_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer = tokenizer,
    callbacks=[early_stopping, training_logs]
)

print("training")
trainer.train()


if args.save != None:
    trainer.model.save_pretrained(f"../models/{args.save}")
    print("saved model")


eval_results = trainer.evaluate(dataset["test"])

pprint(eval_results)

print('Accuracy:', eval_results['eval_accuracy'])

