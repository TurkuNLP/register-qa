from difflib import SequenceMatcher
import re
import numpy as np
import seqeval
from seqeval.metrics import classification_report as seqclassification_report 
from seqeval.metrics import accuracy_score as seqaccuracy_score
import json
import ast
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from transformers import AutoTokenizer
#from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
#from sklearn.metrics import precision_recall_fscore_support

USE_SEQUENCE_MATCHER = True
CUTOFF_FOR_SQM = 0.6 # percentage, 1 = exact match, <0.5 allows for mid span errors
IAA = False
use_tokenizer = True

OUTPUT_PREFIX = "/scratch/project_2002026/amanda/register-qa/evaluation/final_3_results/results/"
INPUT_PREFIX = "/scratch/project_2002026/amanda/register-qa/evaluation/final_3_results/predictions/"


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--input', metavar='FILE', required=True,
                    help='File to be predicted')
    ap.add_argument('--input2', metavar='FILE', required=False,
                    help='File to be predicted (given if --input_format ="annotation"')
    ap.add_argument('--input_format', default="prediction",
                    help='prediction or annotation')
    ap.add_argument('--IAA', type=bool, default=False,
                    help='Calculate using IAA or assuming first input is true, causes minor difference.')
    ap.add_argument('--tokenizer_path', default="xlm-roberta-base",
                    help='Pretrained model name, to download the tokenizer')
    ap.add_argument('--ouput', default="",
                    help='where to output')
    return ap


#..................................F1 calculations..................................#

def measure_f1(a, b):
    """ Measure F1 as 2TP/(2TP + FN + FP) (formula equivalent with the more commonly
      used F1 formula). TP = both annotated, FN, FP = only one person annotated. """
    if a == b:                # the same annotation, F1=1
        return 1
    else:                     # F1 as overlap
        TP = len(np.intersect1d(a, b))
        FP = len(np.setdiff1d(a, b))
        FN = len(np.setdiff1d(b, a))
        return 2*TP/(2*TP+FP+FN)

def parse_result_matrix(M):
    """ Given mxn matrix, calculate rows and columns of zero, and mean of the rest,
      this yields the number of missing (0-rows/columns, one dimension for one annotator)
      and the mean F1 of results that were in agreement """

    # if identity, every one is correct => return F1 =1
    if (M.shape[0] == M.shape[1]) and np.allclose(M, np.eye(M.shape[0])):
        return (1,0,1)      # F1 = 1 and none missed
    elif (np.allclose(M, np.zeros(M.shape))):
        if IAA:    # if evaluating IAA, then sum
            return (0, M.shape[0]+M.shape[1],0)    # F1 = 0 and missed dim1 x dim2 questions
        else:      # only consider the ones in annot 1 to be correct
            return (0, M.shape[0], 0)
    else:
        rows = len(np.where(~M.any(axis=1))[0])   #missed by annot2
        columns = len(np.where(~M.any(axis=0))[0])    #missed by annot1
        mean = np.mean(M[np.where(M>0)])         #mean of non-missed
        # weight the mean with 1-(missed/all)
        mean_with_misses = mean*(1-(rows+columns)/(rows+columns + len(M[np.where(M>0)])))
        return mean, rows+columns, mean_with_misses

def calculate_and_parse_f1(annot1, annot2):
    # MICRO (with flattened array)
    micro = measure_f1(flatten(annot1),flatten(annot2))
    # MACRO
    result_cross= np.zeros((len(annot1), len(annot2)))
    for i, values1 in enumerate(annot1):
        for j, values2 in enumerate(annot2):
            f1 = measure_f1(values1, values2)
            result_cross[i,j] = f1
    macro_without_misses, nulls, macro_with_misses = parse_result_matrix(result_cross)

    return [micro, macro_with_misses, macro_without_misses, nulls]

#..................................vectorisation..................................#

def find_sub_list(sl,l):
    """ From stackexchange, find indices of a sublist in a list (seqmatch added) """
    sll=len(sl)
    if USE_SEQUENCE_MATCHER:      # save the longest match
        seq = SequenceMatcher(None, sl, l)
        i,j,k = seq.find_longest_match()
        if k >= CUTOFF_FOR_SQM*len(sl):
            return [x for x in range(j,j+k)]
    else:
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
                return [x for x in range(ind,ind+sll)]
    return []


def flatten(l):
    return [item for subl in l for item in subl]

def unnest_list(a):
    if a == []:
        return a
    result = [[]]
    for i in a:
        if result[-1] == []:
            result[-1].append(i)
        elif result[-1][-1] == i-1:
            result[-1].append(i)
        else:
            result.append([i])
    return result



def vectorize_annotations(text, questions, answers, punct = True):
    """ Change the questions and answers to a list of indices in the original text.
      This function messes up \n annotations sometimes """

    # find indices
    if not punct:   #split by white space
        splitted_text = text.split()
        indices_questions = [find_sub_list(i.split(), splitted_text) for i in questions]
        indices_answers = [find_sub_list(i.split(), splitted_text) for i in answers]
    else:           # split by white space and punctuation
        splitted_text = re.findall(r'\w+|[^\s\w]+', text)
        indices_questions = [find_sub_list(re.findall(r'\w+|[^\s\w]+', i), splitted_text) for i in questions]
        indices_answers = [find_sub_list(re.findall(r'\w+|[^\s\w]+', i), splitted_text) for i in answers]
  
    vect_text = np.empty(len(splitted_text), dtype=str)
    vect_text[:] = "O"
    vect_text[flatten(indices_questions)] ="Q"
    vect_text[flatten(indices_answers)]="A"

    # tolist() to make seqeval work, removing empty lists from the indices (this was the bug I was struggling with)
    return splitted_text, vect_text.tolist(), [i for i in indices_questions if i!=[]], [i for i in indices_answers if i!=[]]



def tokenize_and_align_labels(tokens, tags, tokenizer):
    """
    Tokenizes and rearranges the labels, i.e.
    [Hi, is, this, yours?]  => [CLS, "Hi", ",", "is", "this", "yours", "?", CLS] 
    [ O,  Q,    Q,      Q]     [  O,   O,   O,    Q,     Q,       Q,    Q,   O ]
    Special tokens mapped to "O"==empty later.
    """
    # adapted from https://huggingface.co/docs/transformers/custom_datasets#tok_ner

    tokenized = tokenizer(
        tokens,
        return_overflowing_tokens=True,
        is_split_into_words=True,
        )
    tags = tags
    word_ids = tokenized.word_ids()
    labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:    # special token
            labels.append("O")
        # Set the label for the first token of each word normally.
        elif word_idx != previous_word_idx:
            labels.append(tags[word_idx])
        else:
            #labels.append(-100)    # Saving the same label for whole word instead
            labels.append(labels[-1])  # for continuation of a word, append the same as previous
        previous_word_idx = word_idx
    tokenized["input_ids"] = tokenized["input_ids"][0]   #unnest bc of return_overflowing_tokens=True
    assert len(tokenized["input_ids"])==len(labels)
    return tokenized, labels



#.................................evaluate.................................#

def prettyprint(prefix, results, return_values=False):
    micro = results[0]
    macro = results[1]
    macro_no_misses = results[2]
    misses = results[3]
    
    if return_values:
        return f'{prefix}\
            \n\tmicro F1 overlap: {micro}\
            \n\tmacro F1 overlap: {macro}\
            \n\tmacro, only found instances: {macro_no_misses}\
            \n\tmisses (per doc): {misses}'
    
    
    print(f'{prefix}\
            \n\tmicro F1 overlap: {micro}\
            \n\tmacro F1 overlap: {macro}\
            \n\tmacro, only found instances: {macro_no_misses}\
            \n\tmisses (per doc): {misses}')

def evaluate_prediction(trues, predictions, options, use_label_vec=False):
    """
    trues = [{text: "words", q: ["...","..."], a: ["...","..."]}, ... ]
    OR if uselabel_vec= True
    trues = [{text: ["word", "word",...], vec = [[label, label,...]]}, ...]
    labels can be Q, B-Q, I-Q, A, B-A, I-A, or O
    NOTE: pred assumed to be double nested!
    """
    # same number of instances in true and pred
    assert len(trues) == len(predictions)
    
    collect_true_vectorized = []
    collect_pred_vectorized = []
    collect_f1_results = {'q':[], 'a':[], 'qa':[]}
       
    
    if use_label_vec:  # assume tags present => format given by the model when doing predictions
        for true, pred in zip(trues, predictions):
            tv = true["labels"]    #true vector
            pv = pred["preds"]   #predicted vector
            # inverse vectorisation
            q1 = unnest_list([i for i, x in enumerate(tv) if x in ["Q","I-Q","B-Q"]])
            a1 = unnest_list([i for i, x in enumerate(tv) if x in ["A","I-A","B-A"]])
            q2 = unnest_list([i for i, x in enumerate(pv) if x in ["Q","I-Q","B-Q"]])
            a2 = unnest_list([i for i, x in enumerate(pv) if x in ["A","I-A","B-A"]])
            # save, in the case of overlap f1 calculate right away
            collect_true_vectorized.append(tv)
            collect_pred_vectorized.append(pv)
            collect_f1_results['q'].append(calculate_and_parse_f1(q1, q2))
            collect_f1_results['a'].append(calculate_and_parse_f1(a1, a2))
            collect_f1_results['qa'].append(calculate_and_parse_f1(q1+a1, q2+a2))
            
    else:   #assume questions given separately => annotated format
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        for true, pred in zip(trues, predictions):
            t = true["text"]
            t_q = true["q"]
            t_a = true["a"]
            p_q = pred["q"]
            p_a = pred["a"]
            
            # vectorize
            st,v1,q1,a1 = vectorize_annotations(t, t_q, t_a)
            st,v2,q2,a2 = vectorize_annotations(t, p_q, p_a)
            
            # realign
            _,tv = tokenize_and_align_labels(st,v1, tokenizer)
            _,pv = tokenize_and_align_labels(st,v2, tokenizer)
            
            if options.use_tokenizer:
                # inverse vectorisation
                q1 = unnest_list([i for i, x in enumerate(tv) if x in ["Q","I-Q","B-Q"]])
                a1 = unnest_list([i for i, x in enumerate(tv) if x in ["A","I-A","B-A"]])
                q2 = unnest_list([i for i, x in enumerate(pv) if x in ["Q","I-Q","B-Q"]])
                a2 = unnest_list([i for i, x in enumerate(pv) if x in ["A","I-A","B-A"]])
                # save, in the case of overlap f1 calculate right away
                collect_true_vectorized.append(tv)
                collect_pred_vectorized.append(pv)
                collect_f1_results['q'].append(calculate_and_parse_f1(q1, q2))
                collect_f1_results['a'].append(calculate_and_parse_f1(a1, a2))
                collect_f1_results['qa'].append(calculate_and_parse_f1(q1+a1, q2+a2))
            
            else:
                # use the vectors untouched by Roberta....
                collect_true_vectorized.append(v1)
                collect_pred_vectorized.append(v2)
                collect_f1_results['q'].append(calculate_and_parse_f1(q1, q2))
                collect_f1_results['a'].append(calculate_and_parse_f1(a1, a2))
                collect_f1_results['qa'].append(calculate_and_parse_f1(q1+a1, q2+a2))
    
    
    with open(options.output, 'w') as file:
        file.write("Metrics from seqeval (whole text, also non-annotated):\n")
        file.write(seqclassification_report(collect_true_vectorized,collect_pred_vectorized))
        file.write("\naccuracy: "+str(seqaccuracy_score(collect_true_vectorized,collect_pred_vectorized)))
        #file.write("\nMetrics from sklearn:\n")
        #file.write(classification_report(collect_true_vectorized, collect_pred_vectorized))
        #file.write("precision, recall, f1:"+str(precision_recall_fscore_support(collect_true_vectorized,collect_pred_vectorized, average='micro')))
        #file.write("\nAccuracy:")
        #file.write(accuracy_score(collect_true_vectorized, collect_pred_vectorized))
        file.write("\nOverlap F1 results:")
        file.write("\n"+str(prettyprint("questions: ", np.mean(collect_f1_results['q'], axis=0),return_values=True)))
        file.write("\n"+str(prettyprint("answers: ", np.mean(collect_f1_results['a'], axis=0),return_values=True)))
    
    
    print("Metrics from seqeval (whole text, also non-annotated):")
    print(seqclassification_report(collect_true_vectorized,collect_pred_vectorized))
    print(options.output)
    print("accuracy: ",seqaccuracy_score(collect_true_vectorized,collect_pred_vectorized))
    print("")
    
    
    print("\nOverlap F1 results:")
    prettyprint("questions: ", np.mean(collect_f1_results['q'], axis=0))
    prettyprint("answers: ", np.mean(collect_f1_results['a'], axis=0))
    prettyprint("both: ", np.mean(collect_f1_results['qa'], axis=0))
    print("-----------------------------------------------------------------------------------")        
        
        
#...............................reading the data...........................# 


def read_annotated_file(path, key):
    data = []
    
    with open(path, "r") as f:
        for line in f:
            d = json.loads(line)
            result = {"text":d["text_plain"], "q":[], "a": []}
            val = d[key]
            if not val:
                continue
            for el in val:
                if "q" in el:
                    result["q"].append(el["q"])
                elif "a" in el:
                    result["a"].append(el["a"])
            data.append(result)
    return data



def wrap_id2label(id2label):
    def id2label_f(d):
        d["labels"] = [id2label[i] for i in d["tags"]]
        return d
    return id2label_f


def read_dataset_file(path, key="test", id2label = {0:"Q", 1:"A", 2 :"O"}):
    dataset = load_from_disk(path)
    data = dataset[key].map(wrap_id2label(id2label))
    return data.select_columns(["tokens", "labels"])
    

def read_json_file(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(ast.literal_eval(line))
    return data
        

def main():   
    options = argparser().parse_args(sys.argv[1:])
    path = INPUT_PREFIX+options.input
    output = OUTPUT_PREFIX+options.input[:-6]+"_results.txt"
    print("options",options.IAA)
    IAA = options.IAA    # this affect the "everythin was missed" calculation. IAA=True considers everything as missed, False ignores preds
    options.output = OUTPUT_PREFIX+options.input[:-6]+"_results_IAA_"+str(IAA)+".txt"
    
    
    if options.input_format == "prediction":
        data_true = read_json_file(path)
    elif options.input_format == "annotation":
        data_true = read_annotated_file(path, "text")
        data_pred = read_annotated_file(options.input2, "text")
    else:
        print("Cannot calculate due to data format!")
 
    if "labels" in data_true[0]:
        evaluate_prediction(data_true, data_true, options, use_label_vec=True)
    elif "q" in data_true[0]:
        evaluate_prediction(data_true, data_pred, options, use_label_vec=False)
    else:
        print("Something wrong with the data format.")
        
        
        
if __name__=="__main__":
    print("alussa", IAA)
    main()
    
    
    
    #path = "/scratch/project_2002026/amanda/register-qa/data/"
    #path = "/scratch/project_2002026/amanda/register-qa/output_fi_gpt_predictions.jsonl"
    #path = "/scratch/project_2002026/amanda/register-qa/test_formatted.jsonl"
    
    #data_true = read_annotated_file(path, "text")
    #data_pred = read_annotated_file(path+"data_fi-Copy1.jsonl", "text")
