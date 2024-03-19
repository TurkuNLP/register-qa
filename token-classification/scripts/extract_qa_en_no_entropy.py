import transformers
import json
import numpy as np
import sys
from itertools import zip_longest

model_name = "TurkuNLP/xlmr-qa-extraction-en"
tokenizer_name = "xlm-roberta-base"

# parameters
# forbidden: words which, if they are alone, should be removed
forbidden = ["question:", "answer:", "question", "answers", "q:", "a:", "faq"]
# len_limit: together with forbidden, e.g. remove strings that are <= 10 chars AND have forbidden word
# this ensures we remove despite small whitespace differences e.g. "vastaus :\n" vs "vastaus:"
len_limit=13  
# note that these should be different for en! 


def find_sentences(text, tokens):
    """
    This function tokenizes text to sentences and similarly separates the
    prediction results (here confusingly tokens). Sentences are defined as
        - things separated by .;:?!
        - thing separated by newline
        - things starting with uppercase letters
    Whitespace, additional punct is added to the previous sentence.
    E.g. This is how this. sentence would be Tokenized!!\n
        >['This is how this. ', 'sentence would be ', 'Tokenized!!\n']
    """
    stop_chars = ".;:!?\n" # plus .isupper()
    sentences=[]    # collect sentences
    indices=[]      # collect indices od chars in sentences, so that...
    sentence_tokens=[]    # ...we can choose which prediction applies to which sentence
    to_skip=0     # this keeps track which chars we have already added
    for index, char in enumerate(text):  
        if to_skip>0:    # Here we first skip the ones we already added in the while loop
            to_skip-=1
        else:        # we have no trailing results
            if index ==0 or char.isupper():   # NEW sentence (or a name, but we do not care about that)
                if index!=0 and sentences[-1]=="":
                    sentences[-1]+=char
                    indices[-1].append(index)
                else:
                    sentences.append(char)
                    indices.append([index])
                while index+to_skip+1 < len(text) and text[index+to_skip+1] not in " \n"+stop_chars:
                    sentences[-1] += text[index+to_skip+1]
                    indices[-1].append(index+to_skip+1)
                    to_skip+=1
            elif char in stop_chars:    # we have a puntuation or new line
                sentences[-1] += char
                indices[-1].append(index)
                while index+to_skip+1 < len(text) and text[index+to_skip+1] in " \n"+stop_chars:
                    sentences[-1] += text[index+to_skip+1]
                    indices[-1].append(index+to_skip+1)
                    to_skip+=1
                sentences.append("")
                indices.append([])
            else:   # regular word separated by whitespace
                sentences[-1] += char
                indices[-1].append(index)
                while index+to_skip+1 < len(text) and text[index+to_skip+1] not in " \n"+stop_chars and not text[index+to_skip+1].isupper():
                    sentences[-1] += text[index+to_skip+1]
                    indices[-1].append(index+to_skip+1)
                    to_skip+=1
    # get the predictions based on indices:
    sentences = [s for s in sentences if s!='']   # this happens in cases where ". ." or "? ?" or end of sentence
    indices = [i for i in indices if i!=[]]   # same
    for values in indices:
        # select correct predictions for each sentence
        sentence_tokens.append([e for e in tokens if e["start"]>= values[0] and e["end"]-1 <= values[-1]])
            
    return sentences, sentence_tokens




def find_sentences_with_separated_numbers(text, tokens):
    debug=False
    """
    This function tokenizes text to sentences and similarly separates the
    prediction results (here confusingly tokens). Sentences are defined as
        - things separated by .;:?!
        - thing separated by newline
        - things starting with uppercase letters
    Whitespace, additional punct is added to the previous sentence.
    E.g. This is how this. sentence would be Tokenized!!\n
        >['This is how this. ', 'sentence would be ', 'Tokenized!!\n']
    """
    stop_chars = ".;:!?\n" # plus .isupper()
    sentences=[]    # collect sentences
    indices=[]      # collect indices od chars in sentences, so that...
    sentence_tokens=[]    # ...we can choose which prediction applies to which sentence
    to_skip=0     # this keeps track which chars we have already added
    for index, char in enumerate(text):
        if debug: print("\n", [char])
        if to_skip>0:    # Here we first skip the ones we already added in the while loop
            if debug: print("added previously")
            to_skip-=1
        else:        # we have no trailing results
            if index ==0 or (char.isupper() or char.isnumeric() or text[index-1].isnumeric()) and char not in stop_chars:   # NEW sentence (or a name, but we do not care about that)
                if debug: print("new entry")
                if index!=0 and sentences[-1]=="":
                    sentences[-1]+=char
                    indices[-1].append(index)
                else:
                    sentences.append(char)
                    indices.append([index])
                while index+to_skip+1 < len(text) and text[index+to_skip+1] not in " \n"+stop_chars or index+to_skip+2 < len(text) and text[index+to_skip+2].isnumeric():
                    sentences[-1] += text[index+to_skip+1]
                    indices[-1].append(index+to_skip+1)
                    to_skip+=1
            elif char in stop_chars:    # we have a puntuation or new line
                if debug: print("stop char")
                sentences[-1] += char
                indices[-1].append(index)
                while index+to_skip+1 < len(text) and text[index+to_skip+1] in " \n"+stop_chars:
                    sentences[-1] += text[index+to_skip+1]
                    indices[-1].append(index+to_skip+1)
                    to_skip+=1
                sentences.append("")
                indices.append([])
            else:   # regular word separated by whitespace
                if debug: print("added as regular")
                sentences[-1] += char
                indices[-1].append(index)
                while index+to_skip+1 < len(text) and text[index+to_skip+1] not in " \n"+stop_chars and not text[index+to_skip+1].isupper() and not text[index+to_skip+1].isnumeric():
                    sentences[-1] += text[index+to_skip+1]
                    indices[-1].append(index+to_skip+1)
                    to_skip+=1
    # get the predictions based on indices:
    sentences = [s for s in sentences if s!='']   # this happens in cases where ". ." or "? ?" or end of sentence
    indices = [i for i in indices if i!=[]]   # same
    for values in indices:
        # select correct predictions for each sentence
        sentence_tokens.append([e for e in tokens if e["start"]>= values[0] and e["end"]-1 <= values[-1]])
            
    return sentences, sentence_tokens

# def normalize(v):
#    return v/np.sum(v)


# def entropy(vector):
#    """Calculates information theory entropy from a vector"""
#    H=0
#    for v in vector:
#        if v != 0:
#            H += v*np.log(1.0/v)
#    return H


def passable(sentence):
    if sentence.lower() in forbidden or any([f in sentence.lower() for f in forbidden]) and len(sentence)<=len_limit:
        # mark it directly as "O"
        return False
    else:
         return True



def extract(texts, tokens,debug=False):
    questions = []
    answers = []
    all_found = []
    # loop over documents
    for te,to in zip(texts,tokens):
        if debug: print("Original text:\n",te, "\n")
        questions.append([])
        answers.append([])
        all_found.append([])
        q_index = 0
        a_index = -1  
        # current: to save correctly, we need to know the prediction that happened in the last roundis none at the start of the text
        current=None    # none at the start of the text
        last=None   # same as current but for saving qa-pairs; we need to know TWO previous predictions
        
        # tokenize to sentence level => we want to keep full sentences.
        sentences, sentence_tokens = find_sentences(te,to)
        if debug: print("Sentences: \n",sentences, "\n")

        # start to loop over sentences
        for s,st in zip(sentences, sentence_tokens):
            if debug: print(s)
            if debug: print(st)
            if st == []:     # We could not tokenize this, either truncation or a a split up date
                if current != "O":
                    all_found[-1].append({"t":s})
                    current = "O"
                else:
                    all_found[-1][-1]["t"] += s
                    current="O"
                continue
            # Init the winning label (this could be clearer, but alas)
            winner = (None,0)
            winner_scores=[0,0,0]
            # find the most probable label for the sentence == winner
            for type_index, type in enumerate(["O","QUESTION","ANSWER"]):
                # vote for the winner; calculate mean score for each weighted by tokens
                entities = [int(i["entity"]==type) for i in st]
                scores = [i["score"] for i in st]
                type_score = np.sum(np.array(entities)*np.array(scores))/len(st)
                if debug: print(type, ": ", type_score)
                winner_scores[type_index] = type_score
                if type_score> winner[1]:
                    winner = (type, type_score)
            # change the winning label manually for problem cases:
            #entropy_of_solution = entropy(normalize(winner_scores))
            #if entropy_of_solution > entropy_limit:       # uncertain prediction
                #if debug: print("Discarded for uncertainty\n")
                #print("Discarded for uncertainty: ", s)   
                #print("Entropy: ", entropy_of_solution)
                #winner = ("O",0)
            if not passable(s):       # sentence in unwanted words
                if debug: print("Discarded for unwanted content\n")
                winner = ("O",0)
            if debug: print("winner: ", winner[0], "\n-------------")
                
            # save as a new index if first or change in type, else concatenate
            # this is complicated as there are to flags:
            # current to tell which class was the last sentence assigned to
            # last to tell if we need to concat or append to question/answer lists

            if current==None or winner[0] != current:
                if winner[0] == "QUESTION":
                    if debug: print("Added as new entry to question")
                    all_found[-1].append({"q"+str(q_index):s})
                    a_index+=1
                    if last == winner[0]:
                        questions[-1][-1] += s
                    else:
                        questions[-1].append(s)
                    last = "QUESTION"
                elif winner[0] == "ANSWER":
                    if debug: print("Added as new entry to answer")
                    all_found[-1].append({"a"+str(a_index):s})
                    if a_index != -1:    # Special case here: we do not want answers that were predicted before any questions;
                        q_index+=1       # otherwise the pairs do not match up: we assume questions come before answers
                        if last == winner[0]:
                            answers[-1][-1] += s
                        else:
                            answers[-1].append(s)
                    last = "ANSWER"
                elif winner[0] == "O":
                    if debug: print("Added as new entry to None")
                    all_found[-1].append({"t":s})
            else:
                if winner[0] == "QUESTION":
                    if debug: print("Continued question")
                    all_found[-1][-1]["q"+str(q_index)] += s
                    questions[-1][-1] += s
                elif winner[0] == "ANSWER":
                    if debug: print("Continued answer")
                    all_found[-1][-1]["a"+str(a_index)] += s
                    if a_index != -1:    # Special case; see similar above for explanation
                        answers[-1][-1] += s
                elif winner[0] == "O":
                    if debug: print("Continued None")
                    all_found[-1][-1]["t"] += s
            current=winner[0]
            if debug: print("current is ", current, "\n")
            #if debug: print(all_found[-1])

        # clean each of the found thingies :) == remove trailing newlines or hyphens (from lists)
        questions[-1] = [q.rstrip(" \n-").lstrip(" \n-") for q in questions[-1]]
        answers[-1] = [a.rstrip(" \n-").lstrip(" \n-") for a in answers[-1]]
        
        
    return all_found, questions, answers


def main(data):
    texts = []
    ids = []
    if ".jsonl" in str(data):
        with open(data) as f:
            for line in f:
                j = json.loads(line)
                texts.append(str(j["text"]).replace("\\n", "\n"))
                ids.append(j["id"])
    elif ".tsv" in str(data):
        with open(data) as f:
            for line in f:
                ids.append(line.split("\t")[0])
                texts.append(line.split("\t")[2])
    else:
        id = 0
        with open(data) as f:
            for line in f:
                texts.append(line.replace("\\n", "\n"))
                ids.append(id)
                id+=1
    
    token_classifier = transformers.pipeline("ner", model=model_name, tokenizer=tokenizer_name, aggregation_strategy=None, ignore_labels=[""],device=0)
    tokens = []  # actually, predictions
    for i,t in enumerate(texts):
        tokens.append(token_classifier(t))

    extracted_texts, questions, answers = extract(texts, tokens)
    
    with open("testi_en.jsonl", "w") as f:
        for id, text, extr in zip(ids, texts, extracted_texts):
            #print(json.dumps({"id": id, "text": text, "extracted": {"qa":e for e in extr}}))
            f.write(json.dumps({"id": id, "text": text, "extracted": extr})+"\n")

    for i in range(len(extracted_texts)):
        print(ids[i])
        for q,a in zip_longest(questions[i], answers[i]):
            
            print("-Question-")
            if q != None:
                print(q.replace("\n", "\\"))
            print("-Answer-")
            if a != None:
                print(a.replace("\n", "\\"))
            print("--")
        print("-----------------------------------------------------------------")

    

        

if __name__ == "__main__":
    data = sys.argv[1]
    main(data)
