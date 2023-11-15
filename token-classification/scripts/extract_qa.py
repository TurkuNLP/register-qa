import json
import sys
import re

# SHOULD MAYBE ADD THE START AND END MARKERS BACK? IF THERE CAN TECHNICALLY BE TWO DIFFERENT ANSWERS BACK TO BACK

#Fi parsebank with enfigpt-fi A: 102272 Q: 75085   from 30896 documents
# parsebank figpt-fi A: 90758 Q: 100745 from 30857 documents
# when discarding 
#amount of docs: 31655   of docs after discarding: 14881   amount of pairs after discarding: 20123
# after taking out the sampled ones
#amount of docs: 31655   amount of docs after discarding: 14080 amount of pairs after discarding: 19081 
# after amandas suggestions
# amount of docs after discarding: 16911 amount of pairs after discarding: 24033


# mc4 enfigpt-fi A: 308477 Q: 201509 from 65713 documents
# mc4 figpt-fi A: 233022 Q: 250837 from 65774 documents
# when discarding
#amount of docs: 66134  amount of docs after discarding: 22927 amount of pairs after discarding: 39721
# after taking out the sampled ones
#amount of docs: 66134  amount of docs after discarding: 22380  amount of pairs after discarding: 38777 
# after amandas suggestions
# amount of docs after discarding: 33271 amount of pairs after discarding: 56454


# cc-fi enfigpt A: 535563 Q: 368970 from 211920 documents
#amount of docs: 211920  amount of docs after discarding: 55746 amount of pairs after discarding: 81116 
# after taking out the sampled ones
#amount of docs: 211920   amount of docs after discarding: 55678  amount of pairs after discarding: 81022 
# after amandas suggestions
#amount of docs after discarding: 75630 amount of pairs after discarding: 114145


# falcon figpt-en A: 234605 Q: 208257 from 81550 documents
# when discarding
#amount of docs: 82259  amount of docs after discarding: 17972  amount of pairs after discarding: 48269 
# after taking out the sampled ones
#amount of docs: 82259  amount of docs after discarding: 17951 amount of pairs after discarding: 48226  
# after amandas suggestions
#amount of docs after discarding: 27853 amount of pairs after discarding: 72128

data = "../results/final/falcon_all.jsonl"


# read the file
with open(data, 'r') as f:
    lines = f.readlines()

#print(lines[0])

ids = [json.loads(line)["id"] for line in lines]
texts = [json.loads(line)["text"] for line in lines]


# take out all the sampled comments by their id
with open("../samples/all_samples.jsonl", 'r') as f:
    sample_lines = f.readlines()

sample_ids = [json.loads(line)["id"] for line in sample_lines]

final_texts = []
final_ids = []
for idd, text in zip(ids, texts):
    if idd in sample_ids:
        continue
    else:
        final_ids.append(idd)
        final_texts.append(text)



questions = []
answers = []
junked = []
# loop through the qa to take the pairs
for qa in final_texts:
    temp_q = []
    temp_a = []
    junk = []
    short_count = 0
    previous = ""
    for part in qa: # loop through the list 
        if type(part) is dict:
            if "q" in part:
                if len(part["q"]) < 15: # if a short part
                    if previous == "q":
                        temp_q[len(temp_q) -1] = temp_q[len(temp_q) -1] + part["q"]
                    else:
                        short_count = short_count + 1
                else:
                    if previous == "q":
                        temp_q[len(temp_q) -1] = temp_q[len(temp_q) -1] + part["q"]
                    else:
                        previous = "q"
                        temp_q.append(part["q"])
            elif "a" in part:
                if len(part["a"]) < 15: # if a short part
                     #check that there is at least one question before (cannot start with answer)
                     # and concat the strings if previous was answer as well
                    if previous == "a" and temp_q:
                        temp_a[len(temp_a) -1] = temp_a[len(temp_a) -1] + part["a"]
                    elif temp_q:
                        short_count = short_count + 1

                else:
                    if previous == "a":
                        temp_a[len(temp_a) -1] = temp_a[len(temp_a) -1] + part["a"]
                    elif not temp_q:
                        # if there is no question before an answer
                        continue
                    else:
                        previous = "a"
                        temp_a.append(part["a"])

            # this part below is kind of optional, but maybe good to use?
            # Amanda can modify the whole script if needed
           
            elif "t" in part:
                if len(part["t"]) < 5: 
                    # if there is a reaaally short one
                    if previous == "a":
                        temp_a[len(temp_a) -1] = temp_a[len(temp_a) -1] + part["t"]
                    elif previous == "q":
                        temp_q[len(temp_q) -1] = temp_q[len(temp_q) -1] + part["t"]
                    # else:
                    #     short_count = short_count + 1

    # if count is too big discard the document (but add to keep lenght of all)
    if short_count > 3:
        questions.append([])
        answers.append([])
        #junk.append([])
    else:
        questions.append(temp_q)
        answers.append(temp_a)
        #junked.append(junk)


# save in the format
multiple=open("../qa_predicted_final_files/falcon_qa.jsonl","wt")
pair_count = 0
doc_count = 0
for idd, que,ans in zip(final_ids, questions, answers):
    #print(text_id)
    save = {}
    tmp = {}
    # check that there is at least one pair
    if len(que) != 0 and len(ans) != 0: 
        #and len(ans) == len(que)
        doc_count = doc_count + 1
        for i, (q, a) in enumerate(zip(que,ans)):
            pair_count = pair_count + 1 
            tmp[f"q{i}"] = q
            tmp[f"a{i}"] = a
        save["id"] = idd
        save["qa"] = tmp
        line=json.dumps(save,ensure_ascii=False)
        print(line,file=multiple)
    else:
        continue

print("amount of docs:", len(ids))
print("amount of docs after discarding:", doc_count)
print("amount of pairs after discarding:", pair_count)