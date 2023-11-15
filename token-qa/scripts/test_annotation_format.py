import json
import sys
import datasets
import re

data = "../samples/sample-falcon/annotated/formatted/en_dev_formatted.jsonl"

# SHOULD MAYBE ADD THE START AND END MARKERS BACK? IF THERE CAN TECHNICALLY BE TWO DIFFERENT ANSWERS BACK TO BACK


# read the file
with open(data, 'r') as f:
    lines = f.readlines()

#print(lines[0])

# THIS FOR CHATGPT FORMAT
# texts = [json.loads(line)["text_chatgpt"] for line in lines]
# plain_texts = [json.loads(line)["text"] for line in lines]

# THIS FOR ANNOTATED FORMAT
texts = [json.loads(line)["text"] for line in lines]
plain_texts = [json.loads(line)["text_plain"] for line in lines]

def clean(text):
    # '''
    # Just a helper fuction to add a space before the punctuations for better tokenization
    # '''
    # filters = ["!", "#", "$", "%", "&", "(", ")", "/", "*", ".", ":", ";", "<", "=", ">", "?", "@", "[",
    #            "\\", "]", "_", "`", "{", "}", "~", "'", ","]
    # for i in text:
    #     if i in filters:
    #         text = text.replace(i, " " + i)

    text = re.findall(r'\w+|[^\s\w]+', text)
    #print(text)
            
    return text


ids = [json.loads(line)["id"] for line in lines]


questions = []
answers = []
junked = []
tagged = []
splits = []
q_count = 0
a_count = 0
# loop through the qa to take the pairs back
for qa in texts:
    tags = []
    temp_q = []
    temp_a = []
    junk = []
    splitted = []
    temp_split = []
    if type(qa) is str:
        print("NOT ANNOTATED")
        print(qa)
        # here could add the splitting and put O as label everywhere
        if qa.isspace(): # if there is only whitespaces (\n as well)
            split = [qa] 
        else:
            qa = clean(qa) # added "cleaning"
            split = clean(qa)
            #split = qa.split()
        splitted.append(split)
        for one in split:
            tags.append("O")
        junk.append(qa)
    else:
        for part in qa: # loop through the list
            if type(part) is dict:
                if "q" in part:
                    #part["q"] = clean(part["q"]) # added "cleaning"
                    split= clean(part["q"])
                    #split = part["q"].split()
                    splitted.append(split)
                    for one in split:
                        tags.append("QUESTION")
                    temp_q.append(part["q"])
                    q_count += 1
                elif "a" in part:
                    #part["a"] = clean(part["a"]) # added "cleaning"
                    split = clean(part["a"])
                    #split = part["a"].split()
                    splitted.append(split)
                    for one in split:
                        tags.append("ANSWER")
                    temp_a.append(part["a"])
                    a_count += 1
                elif "t" in part:
                    #part["t"] = clean(part["t"]) # added "cleaning"
                    split = clean(part["t"]) 
                    #split = part["t"].split()
                    splitted.append(split)
                    for one in split:
                        tags.append("O")
                    junk.append(part["t"])
            # else:
            #     junk.append(part)
            #     if part.isspace(): # if there is only whitespaces (\n as well)
            #         split = [part]
            #     else:
            #         split = part.split()
            #     splitted.append(split)
            #     for one in split:
            #         tags.append("O")

        # for lista in qa: # -> now only one list! so no need to loop but works like this too :D
        #     if type(lista) is str:
        #         if lista.isspace(): # if there is only whitespaces (\n as well)
        #             split = [lista]
        #         else:
        #             split = lista.split()
        #         splitted.append(split)
        #         for one in split:
        #             tags.append("O")
        #         junk.append(lista)
        #     else:
        #         for part in lista:
        #             #print(part)
        #             if type(part) is dict:
        #                 if "q" in part:
        #                     split = part["q"].split()
        #                     splitted.append(split)
        #                     for one in split:
        #                         tags.append("QUESTION")
        #                     temp_q.append(part["q"])
        #                 elif "a" in part:
        #                     split = part["a"].split()
        #                     splitted.append(split)
        #                     for one in split:
        #                         tags.append("ANSWER")
        #                     temp_a.append(part["a"])
        #             else:
        #                 junk.append(part)
        #                 if part.isspace(): # if there is only whitespaces (\n as well)
        #                     split = [part]
        #                 else:
        #                     split = part.split()
        #                 splitted.append(split)
        #                 for one in split:
        #                     tags.append("O")


    questions.append(temp_q)
    answers.append(temp_a)
    junked.append(junk)
    tagged.append(tags)
    for one in splitted:
        #print(one)
        temp_split = temp_split + one
    splits.append(temp_split)

print("A:", a_count)
print("Q:", q_count)
print("------")
print(tagged[2], splits[2])


## save ids, tags and splitted texts to a list of dictionaries -> then load as dataset? maybe easiest?
dataset = {}
dataset["id"] = ids
dataset["tags"] = tagged
dataset["tokens"] = splits
dataset["text"] = plain_texts

tags = datasets.ClassLabel(num_classes=3, names=['QUESTION', 'ANSWER', 'O'])
# create dataset
save = datasets.Dataset.from_dict(mapping=dataset, features=datasets.Features({"text": datasets.Value(dtype='string'),"tags":datasets.Sequence(tags), 'id': datasets.Value(dtype='string'), 'tokens': datasets.Sequence(feature=datasets.Value(dtype='string'))}))

#save = Dataset.from_dict(dataset_list)

print(save)
print(save[0])
print(save.features["tags"].feature.int2str(0))


# save the dataset
#save.save_to_disk("../qa-labelled/annotated/dev_annotated_dataset")
save.to_json("../samples/sample-falcon/annotated/formatted/en_dev_dataset.jsonl", force_ascii=False)