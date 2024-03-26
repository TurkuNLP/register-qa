import json
import sys

data = sys.argv[1]


# read the file
with open(data, 'r') as f:
    lines = f.readlines()

qas = [json.loads(line)["qa"] for line in lines]

ids = [json.loads(line)["id"] for line in lines]


questions = []
answers = []
# loop through the qa to take the pairs back
for qa in qas:
    temp_q = []
    temp_a = []
    i = 0
    while f"q{i}" in qa:
        temp_q.append(qa[f"q{i}"])
        temp_a.append(qa[f"a{i}"])
        i += 1
    
    questions.append(temp_q)
    answers.append(temp_a)

print("examples:")
print("id", ids[0])
print("question:")
print(questions[0][0])
print("answer:")
print(answers[0][0])
print("--------------")

print("from how many documents:")
print("questions:", len(questions))
print("answers:", len(answers))
print("how many in those documents:")

count = 0
for q in questions:
    count += len(q)

print("questions:", count)

count = 0
for a in answers:
    count += len(a)

print("answers:", count)