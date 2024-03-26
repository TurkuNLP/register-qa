import transformers
import json
import numpy as np
import sys
from itertools import zip_longest
from custom_ner_pipe import custom_ner_pipe

model_name = "TurkuNLP/xlmr-qa-extraction-fi"
tokenizer_name = "xlm-roberta-base"

# parameters
# Set the value of entropy to get rid of uncertain predictions
entropy_limit = 0.98   # TODO optimize
# forbidden: words which, if they are alone, should be removed
forbidden = ["vastaus:", "kysymys:", "vastaa", "vastaus", "kysymys"]
# len_limit: together with forbidden, e.g. remove strings that are <= 10 chars AND have forbidden word
# this ensures we remove despite small whitespace differences e.g. "vastaus :\n" vs "vastaus:"
len_limit=10  
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

def normalize(v):
    return v/np.sum(v)


def entropy(vector):
    """Calculates information theory entropy from a vector"""
    H=0
    for v in vector:
        if v != 0:
            H += v*np.log(1.0/v)
    return H


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
            if st == []:     # TODO: tokenisation limit
                # We could not tokenize this far => end
                break
                
            # calculate sentence label as mean of predictions
            sentence_scores = np.array([0.0,0.0,0.0])
            mean_entropy_per_token = 0
            for token in st:
                sentence_scores += token["full_score"]
                mean_entropy_per_token += entropy(normalize(token["full_score"]))
            mean_scores = sentence_scores/len(st)
            mean_entropy_per_token /= len(st)
            id2label = st[0]["full_score_names"]   # this is model.config.id2label
            winner = (id2label[mean_scores.argmax()],mean_scores.max())

            # calculate entropy of the sentence (trying both ways)
            entropy_of_sentence = entropy(normalize(mean_scores))   # normalization actually not needed as the mean is already normalized
          
            if entropy_of_sentence > entropy_limit:       # uncertain prediction
                if debug: print("Discarded for uncertainty\n")
                #print("Discarded for uncertainty: ", s)   # these two lines were used in the entropy limit optimisation
                #print("Entropy per token: ", mean_entropy_per_token)
                #print("Entropy per sentence: ", entropy_of_sentence)
                #print(mean_scores)
                winner = ("O",0)
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
                    all_found[-1].append({"q"+str(q_index):s})
                    a_index+=1
                    if last == winner[0]:
                        questions[-1][-1] += s
                    else:
                        questions[-1].append(s)
                    last = "QUESTION"
                elif winner[0] == "ANSWER":
                    all_found[-1].append({"a"+str(a_index):s})
                    if a_index != -1:    # Special case here: we do not want answers that were predicted before any questions;
                        q_index+=1       # otherwise the pairs do not match up: we assume questions come before answers
                        if last == winner[0]:
                            answers[-1][-1] += s
                        else:
                            answers[-1].append(s)
                    last = "ANSWER"
                elif winner[0] == "O":
                    all_found[-1].append({"t":s})
            else:
                if winner[0] == "QUESTION":
                    all_found[-1][-1]["q"+str(q_index)] += s
                    questions[-1][-1] += s
                elif winner[0] == "ANSWER":
                    all_found[-1][-1]["a"+str(a_index)] += s
                    if a_index != -1:    # Special case; see similar above for explanation
                        answers[-1][-1] += s
                elif winner[0] == "O":
                    all_found[-1][-1]["t"] += s
            current=winner[0]
            if debug: print("current is ", current, "\n")

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
    else:
        id = 0
        with open(data) as f:
            for line in f:
                texts.append(line.replace("\\n", "\n"))
                ids.append(id)
                id+=1

    # define model and tokenizer for custom pipe
    device = 'cuda'
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, truncation=True)
    model = transformers.AutoModelForTokenClassification.from_pretrained(model_name)
    model.to(device)
    token_classifier = custom_ner_pipe(model, tokenizer, device=device)
    
    tokens = []  # actually, predictions per token
    for i,t in enumerate(texts):
        tokens.append(token_classifier.predict(t))   # do prediction to each token

    extracted_texts, questions, answers = extract(texts, tokens, debug=False)

    print(extracted_texts)
    exit()
    with open("extracted.jsonl", "w") as f:
        for id, text, extr in zip(ids, texts, extracted_texts):
            continue
            #print(json.dumps({"id": id, "text": text, "extracted": extr}))
            #f.write(json.dumps({"id": id, "text": text, "extracted": extr}))

    
    for i in range(len(extracted_texts)):
        print(ids[i])
        for q,a in zip_longest(questions[i], answers[i]):
            
            print("-Kysymys-")
            print(q)
            print("-Vastaus-")
            print(a)
            print("--")
        print("-----------------------------------------------------------------")

    

        

if __name__ == "__main__":
    data = sys.argv[1]
    main(data)
