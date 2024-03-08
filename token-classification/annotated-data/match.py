import sys

#[annieske@puhti-login15 sorted]$ zcat sorted_qa_binary_cc-fi_all.tsv.gz sorted_qa_binary_mc4-fi.tsv.gz sorted_qa_binary_parsebank.tsv.gz | python3 ../../annotated-data/match.py > ../../annotated-data/matched.tsv                   

data = sys.stdin.readlines() # the original files
for i in range(len(data)):
    data[i] = data[i].replace("\n", "")
    data[i] = data[i].split("\t") # split at the labels

# the id file
ids = []
 
with open("../../annotated-data/manual_eval_ids.tsv") as f:
  for line in f:
    ids.append(line)

for i in range(len(ids)):
    ids[i] = ids[i].replace("\n", "")
    #data[i] = data[i].split("\t") # split at the labels
 

new = []
new_ids = []

# compare
for one in ids:
    for i in range(len(data)):
        if one == data[i][0]: # in
            new.append(data[i][2])
            new_ids.append(one)
            break


# print to file
final = []
for i in range(len(new)):
    dummy =  new[i] # new_ids[i] + '\t' +
    final.append(dummy)

for i in range(len(final)):
    print(final[i])