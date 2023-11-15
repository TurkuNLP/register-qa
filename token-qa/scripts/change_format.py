import ast
import glob
import json
import os
path = "/scratch/project_2005092/Anni/qa-register/token-qa/samples/sample-falcon/annotated/og"
# def get_plaintext(annotated):
#     return "".join(["".join(y.values()) for y in annotated["text"]])
def read_original_file(path):
    """ Read a non-nested jsonl file and extract the text """
    texts = {}
    with open(path, 'r') as f:
        for line in list(f):
            texts[ast.literal_eval(line)["id"]] = ast.literal_eval(line)["text"]
    return texts

import re
def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)

def convert(file_contents):
    docs = file_contents.split("\n")
    formatted_docs = []
    for d in docs:
        res = []
        if not d:
            continue
        
        new_d = remove_non_ascii(d)

        a = ast.literal_eval(new_d)
        if type(a["text"]) == str:
            if a["text"]:
                res.append({"t": a["text"]})
        else:
            for p in a["text"]:
                if type(p) == str:
                    if p:
                        res.append({"t": p})
                else:
                    for pp in p:
                        if type(pp) == str:
                            if pp:
                                res.append({"t": pp})
                        else:
                            res.append(pp)
        res_final = {"id": a.get("id"), "text": res}
        formatted_docs.append(res_final)
    return formatted_docs

# read all 
texts = {}
for filename2 in glob.glob(os.path.join("../samples/sample-falcon/originals", "*.jsonl")):
    temp = read_original_file(filename2)
    texts = {**texts, **temp}

for filename in glob.glob(os.path.join(path, "*.jsonl")):
    with open(os.path.join(os.getcwd(), filename), "r") as f_in:
        try:
            file_contents = f_in.read()
        except:
            with open(
                os.path.join(os.getcwd(), filename), "r", encoding="latin-1"
            ) as f:
                file_contents = f_in.read()
        formatted_docs = convert(file_contents)
        print(filename)
        if formatted_docs:
            new_filename = ".".join(filename.split(".")[:-1]) + "_formatted.jsonl"
            print(new_filename)
            with open(os.path.join(os.getcwd(), new_filename), "a") as f_out:
                for doc in formatted_docs:
                    doc["text_plain"] = texts[doc["id"]]
                    f_out.write(f"{json.dumps(doc, ensure_ascii=False)}\n") 