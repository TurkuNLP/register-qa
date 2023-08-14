import sys
import re
import csv
import json
from tqdm import tqdm


tsv_file = sys.argv[1]

# open tsv that has been through NER (can also take og file if wanted so)
# get the question part (and answer part)
data=[]
with open(tsv_file, 'r') as f:
    for line in f:
        line=line.rstrip("\n")
        cols=line.split("\t")
        data.append(cols)
    data = data[1:]

# BECAUSE OF THIS HAS TO HAVE SECOND ARGUMENT BUT IT CAN BE NONSENSE
if sys.argv[2] == "ner":
    questions = [one[1] for one in data]
    ids = [one[0] for one in data]
    answers = [one[2] for one in data]

else:
    texts = [one[2] for one in data]
    ids = [one[0] for one in data]

#questions = questions[:20]

if sys.argv[2] == "ner":
    for i in tqdm(range(len(questions))):
        # regex here, take stuff after these
        # (?:) -> group without capture, now I only capture the match and the thing that comes after
        # (?:(.*)^Kysymys (.*))|(?:(.*)^Kysymys: (.*))
        # if I want the group before the kysymys, I need to add (.*) and take group 2 for the final question

        m = re.search("(.*)(Kysymys:)(.*)",questions[i])
        if m != None:
            questions[i] = m.group(3)
            continue
        m = re.search("(.*)(Kysymys)(.*)",questions[i]) # non-capturing group outside for | makes all none and capturing only takes the outer -> make separate searches...
        if m != None:
            questions[i] = m.group(3)
            continue

        m = re.search("(.*)(Hae kysymyksiä  Hae)(.*)",questions[i]) 
        if m != None:
            questions[i] = m.group(3)
            continue
        m = re.search("(.*)(Usein kysytyt kysymykset)(.*)",questions[i]) 
        if m != None:
            questions[i] = m.group(3)
            continue
        m = re.search("(.*)(Uusimmat kysymykset)(.*)",questions[i]) 
        if m != None:
            questions[i] = m.group(3)
            continue

            #Kysymykset, vastaukset ja kommentit

            # there seem to be so many variations that this is getting impossible / there will be countless of different things...
            # would it be better to annotate some and possibly make a better NER model?
        
        # empty and do not write to file later
        m = re.search("(Vastaukset(.*)|Kysymykset(.*))",questions[i]) 
        if m != None:
            questions[i] == ""
            continue

    # then make similar thing for answers!
    for i in tqdm(range(len(answers))):

        m = re.search("(.*)(Tapio Ketonen:)(.*)",answers[i])
        if m != None:
            answers[i] = m.group(3)
            continue
        
        m = re.search("(.*)(Vastaus)(.*)",answers[i])
        if m != None:
            answers[i] = m.group(3)
            continue

        m = re.search("(.*)(Keskustelu)(.*)",answers[i])
        if m != None:
            answers[i] = m.group(1)
            continue
            
            
        m = re.search("(Vastaukset|Kysymykset)",answers[i]) 
        if m != None:
            answers[i] == ""
            continue

else:
    # do heuristics before NER
    #THIS IS VERY SLOW! THE TEXTS ARE TOO LONG I GUESS? OR THE REGEX TRIES TO DO TOO MUCH AT ONCE..
    # splitting it into multiple ones helped!
    questions = []
    answers = []
    not_found = []
    new_ids = []
    #for i in tqdm(range(len(texts))):
        # m = re.search("(.*)(Kysymys)(.*)(Vastaus)(.*)",texts[i]) # non-capturing group outside for | makes all none and capturing only takes the outer -> make separate searches...
        # if m != None:
        #     questions.append(m.group(3))
        #     answers.append(m.group(5))
        #     continue
        
        # m = re.search("(.*)(Kysymys)(.*)(Tapio Ketonen:)(.*)",texts[i]) # non-capturing group outside for | makes all none and capturing only takes the outer -> make separate searches...
        # if m != None:
        #     questions.append(m.group(3))
        #     answers.append(m.group(5))
        #     continue

        # check that Kysymys kuuluu is not deleted ..
        # m = re.search("(.*)(Kysymys kuuluu)(.*)",texts[i]) # non-capturing group outside for | makes all none and capturing only takes the outer -> make separate searches...
        # if m != None:
        #     not_found.append([texts[i], ids[i]])
        #     continue

        # m = re.search("^(.+)(Kysymys|Kysymys:)(.+)",texts[i]) # non-capturing group outside for | makes all none and capturing only takes the outer -> make separate searches...
        # if m == None:
        #     not_found.append([texts[i], ids[i]])
        #     continue
        # a = re.search("^(.+)(Vastaus)(.+)", m.group(3))
        # if a != None:
        #     answers.append(a.group(3))
        #     questions.append(a.group(1))
        #     continue
        # a = re.search("^(.+)(Tapio Ketonen:)(.+)", m.group(3))
        # if a != None:
        #     answers.append(a.group(3))
        #     questions.append(a.group(1))
        #     continue
        
        # if a == None:
        #     answers.append("")
        #     not_found.append([texts[i], ids[i]])

        # SPLITTING
    #     escaped = r"Kysymys:|Kysymys |Kysymys\\"
    #     splitted = re.split(escaped, texts[i])
    #     if len(splitted) == 1 and "Kysymys" in splitted:
    #         if splitted[0].startswith("kuuluu"):
    #             not_found.append([texts[i], ids[i]])
    #             continue
    #         #else:
    #             # TODO figure out what to do if Kysymys: is at beginning and I want the rest ughh

    #         escaped = r"Vastaus:|Vastaus |Vastaus\\"
    #         # not found -> or it is at the beginning
    #         ans_split = re.split(escaped, splitted[0])
    #         if len(ans_split) == 2:
    #             questions.append(ans_split[0])
    #             answers.append(ans_split[1])
    #             new_ids.append(ids[i])
    #     elif len(splitted) >= 2:
    #         if splitted[1].startswith("kuuluu"):
    #             not_found.append([texts[i], ids[i]])
    #             continue
    #         escaped = r"Vastaus:|Vastaus |Vastaus\\"
    #         ans_split = re.split(escaped, splitted[1])
    #         if len(ans_split) == 2:
    #             questions.append(ans_split[0])
    #             answers.append(ans_split[1])
    #             new_ids.append(ids[i])
    #     else:
    #         not_found.append([texts[i], ids[i]])
            
    # print("no change", len(not_found))
    # print("qas", len(questions))

# also make something that can separate multiple questions and answers -> make list of lists? -> would be beneficial to keep them together after all
# Q: A: are something that happen in both english and parsebank

    # TEMPORARY
    not_found = [[text, idd] for text, idd in zip(texts, ids)]
    
    multiples_q = []
    multiples_a = []
    q_temp = []
    a_temp = []
    count = 0
    
    # for english 
    # Q1 : -> Q\d? : |  Explanation :
    # Question: Response:
    # Question: Answer:
    #Q. A.
    # Q. Answer
    # Q #16: A: -> Q #\d+:
    #The Question is : The Answer is :
    # Q A -> this will take other stuff too so might have to skip
    # if there are faqs in falcon then could try separating by \n ?

    for i in tqdm(range(len(not_found))):
        # TRYING SPLITTING
        escaped = r"Q:|Q\d? :|Question:|Q\.|Q #\d+:|The Question is :"
        splitted = re.split(escaped, not_found[i][0])
        # if at the beginning
        if len(splitted) == 1 and "Q" in splitted[0]:
            splitted[0] = re.sub(escaped, '', splitted[0]) # replace the beginning from the thing
            escaped = r"A:|Explanation :|Response:|Answer:|A\."
            ans_split = re.split(escaped, splitted[0])
            if len(ans_split) >= 2:
                for j in range(len(ans_split)):
                    if j % 2 == 0:
                        q_temp.append(ans_split[j])
                    else:
                        a_temp.append(ans_split[j])
                not_found[i][0] = ""
        if len(splitted) == 2:
            escaped = r"A:|Explanation :|Response:|Answer:|A\."
            ans_split = re.split(escaped,splitted[1])
            if len(ans_split) == 2:
                q_temp.append(ans_split[0])
                a_temp.append(ans_split[1])
                not_found[i][0] = ""
                count += 1
                
        elif len(splitted) > 2:
            for j in range(len(splitted)):
                if j == 0:
                    continue
                else:
                    escaped = r"A:|Explanation :|Response:|Answer:|A\."
                    q_a_split = re.split(escaped,splitted[j])
                    if len(q_a_split) == 2:
                        q_temp.append(q_a_split[0])
                        a_temp.append(q_a_split[1])
                        not_found[i][0] = ""
                        count += 1
                        continue

        # TODO how the hell does this not work for everything :D 
        escaped = r"Kysymys:|Kysymys |Kysymys\\"
        splitted = re.split(escaped, not_found[i][0])
        if "kuuluu:" in splitted[0]:
                pass
        elif len(splitted) == 1 and "Kysymys" in splitted[0]:
            splitted[0] = re.sub(escaped, '', splitted[0]) # replace the beginning from the thing
            escaped = r"Vastaus:|Vastaus |Vastaus\\|Tapio Ketonen:"
            ans_split = re.split(escaped, splitted[0])
            if len(ans_split) >= 2:
                for j in range(len(ans_split)):
                    if j % 2 == 0:
                        q_temp.append(ans_split[j])
                    else:
                        a_temp.append(ans_split[j])
                    not_found[i][0] = ""
                    count += 1
                
        
        elif len(splitted) == 2:
            escaped = r"Vastaus:|Vastaus |Vastaus\\|Tapio Ketonen:"
            ans_split = re.split(escaped, splitted[1])
            if len(ans_split) == 2:
                q_temp.append(ans_split[0])
                a_temp.append(ans_split[1])
                not_found[i][0] = ""
                count += 1
                
        elif len(splitted) > 2:
            for j in range(len(splitted)):
                if "kuuluu:" in splitted[j]:
                    continue
                if j == 0:
                    continue
                else:
                    escaped = r"Vastaus:|Vastaus |Vastaus\\|Tapio Ketonen:"
                    q_a_split = re.split(escaped, splitted[j])
                    if len(q_a_split) == 2:
                        q_temp.append(q_a_split[0])
                        a_temp.append(q_a_split[1])
                        not_found[i][0] = ""
                        count += 1
                        continue

        escaped = "vastaa\n\nUusimmat kysymykset"
        splitted = re.split(escaped, not_found[i][0])
        if len(splitted) >= 2:
            print("???")
            q_a_split = splitted[1].split("Tapio Ketonen:")
            if len(q_a_split) >= 2:
                for j in range(len(q_a_split)):
                    if j % 2 == 0:
                        q_temp.append(q_a_split[0])
                    else:
                        a_temp.append(q_a_split[1])
                    count += 1
                not_found[i][0] = ""





        # TODO still takes just the "outer" part? and it's a bit slow so yeah
        # how to make these fail faster :D just check for Q: at first? is that any faster? non-capturing groups is also apparently faster

        # pattern = re.compile(r'(.+)(Q:)?(.+)(A:)?(.+)(Q:)?')
        # matches = pattern.finditer(not_found[i][0])
        # if matches != None:
        #     count = 0
        #     for match in matches:
        #         print("found", match)
        #         count += 1
        #         print(count)
        #         q_temp.append(match.group(3))
        #         a_temp.append(match.group(5)) 
        #         not_found[i][0] = ""
        # else:
        #     pattern = re.compile(r'(.+)(Kysymys)(.+)(Tapio Ketonen:)(.+)')
        #     matches = pattern.finditer(not_found[i][0])
        #     if matches != None:
        #         for match in matches:
        #             q_temp.append(match.group(3))
        #             a_temp.append(match.group(5))
        #             not_found[i][0] = ""
        
            
        multiples_a.append(a_temp)
        multiples_q.append(q_temp)
        a_temp = []
        q_temp = []

    print(count)
        
    # json dictionary, then just put as text: {q1: "", a1 = "", q2 = "" ....} so inside text will be many pairs -> may be annoying to load back if there can be any number of qa stuff inside "text"
    # but it's the best I have right now :D
    multiple=open("../data/results/combined-cc-fi.jsonl","wt")
    #print(len(not_found), len(multiples_q), len(multiples_a))
    for text_id, que,ans in zip(not_found, multiples_q, multiples_a):
        #print(text_id)
        save = {}
        tmp = {}
        if text_id[0] == "" and len(que) != 0 and len(ans) != 0:
            for i, (q, a) in enumerate(zip(que,ans)):
                tmp[f"q{i}"] = q
                tmp[f"a{i}"] = a
            save["id"] = text_id[1]
            save["qa"] = tmp
            line=json.dumps(save,ensure_ascii=False)
            print(line,file=multiple)
        else:
            continue
    
    


    with open("../data/results/not_found_cc-fi.tsv", 'w') as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        header = ["id", "text"]
        writer.writerow(header)
        for t, d in not_found:
            if t == "":
                continue
            line = [d, t]
            writer.writerow(line)


# #save the ones where question and answer are separated be it with ner used before or not
# with open("../data/results/cleaned_parsebank.tsv", 'w') as outfile:
#     writer = csv.writer(outfile, delimiter="\t")
#     header = ["id", "question", "answer"]
#     writer.writerow(header)
#     for d, q, a in zip(new_ids, questions, answers):
#         if q == "" or a == "":
#             continue
#         line = [d, q, a]
#         writer.writerow(line)