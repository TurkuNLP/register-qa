import json
import sys
from tqdm import tqdm

data = sys.argv[1]


# read the file
with open(data, 'r') as f:
    lines = f.readlines()

texts = [json.loads(line)["text"] for line in lines]
ids = [json.loads(line)["id"] for line in lines]


# parent_comment_id, thread_id, quoted_comment_id, “msg_type”: “thread_start”, "msg_type": "comment"
#→ would have to make threads of comments


threads = {}
thread_starts = {}
thread_comments = {}

# read one dictionary at a time
for example in tqdm(lines):
    example = json.loads(example)
    # this means the comment is a response to someone else -> skip
    if example["meta"]["sourcemeta"]["parent_comment_id"] != "0":
        continue
    # this is the most important part
    if example["meta"]["sourcemeta"]["thread_id"] in threads:
        threads[example["meta"]["sourcemeta"]["thread_id"]].append(example)
    else:
        threads[example["meta"]["sourcemeta"]["thread_id"]] = [example]

    # not sure what this means, maybe reply sort of? where there's a quote? not useful for me I don't think
    if "quoted_comment_id" in example["meta"]["sourcemeta"]:
        pass
    # what is the difference between comment_id and id? I guess I will use id or not??
    if "comment_id" in example["meta"]["sourcemeta"]:
        pass
    if "msg_type" in example["meta"]["sourcemeta"]:
        # this I could interpret as a question
        if example["meta"]["sourcemeta"]["msg_type"] == "thread_start":
            # if example["meta"]["sourcemeta"]["thread_id"] in thread_starts:
            #     thread_starts[example["meta"]["sourcemeta"]["thread_id"]].append(example)
            #else:
            thread_starts[example["meta"]["sourcemeta"]["thread_id"]] = example
        # and then these as answers hmmm? then of course would have to use ner or something to actually put these like that
        elif example["meta"]["sourcemeta"]["msg_type"] == "comment":
            if example["meta"]["sourcemeta"]["thread_id"] in thread_comments:
                thread_comments[example["meta"]["sourcemeta"]["thread_id"]].append(example)
            else:
                thread_comments[example["meta"]["sourcemeta"]["thread_id"]] = [example]


print(len(thread_starts))
# there is not as many comments as thread starts here ->
print(len(thread_comments))

pop_time = []
# remove from the starts stuff that does not have a comment/comments
for key in thread_starts:
    if key not in thread_comments:
        pop_time.append(key)

for key in pop_time:
    thread_starts.pop(key)

pop_time = []
# do the same the other way
for key in thread_comments:
    if key not in thread_starts:
        pop_time.append(key)

for key in pop_time:
    thread_comments.pop(key)

print("----")
# now there should be as many, thread comments might have more than one comment per thread id
print(len(thread_starts))
print(len(thread_comments))

count = 0
for key in thread_comments:
    count += len(thread_comments[key])

print("comment count", count)

# print example of start and comment(s)
iterator = iter(thread_starts)
res = next(iterator)
res2 = next(iterator)
print(thread_starts[res2])
print(thread_comments[res2])

# from threads I need to remove comments that have the same author as the thread start
for start in thread_starts:
    pop_time = []
    for i in range(len(thread_comments[start])):
        # haha such a long thing here now :D 
        if thread_starts[start]["meta"]["sourcemeta"]["author"] == thread_comments[start][i]["meta"]["sourcemeta"]["author"] or "aloittaja" in thread_comments[start][i]["meta"]["sourcemeta"]["author"] :
            pop_time.append(i)
    pop_time.reverse()
    for one in pop_time:
        thread_comments[start].pop(one)
        if not thread_comments[start]:
            thread_comments.pop(start)

# do this again, could be a def at this point
for key in thread_starts:
    if key not in thread_comments:
        pop_time.append(key)

for key in pop_time:
    thread_starts.pop(key)

print(len(thread_starts))
# there is not as many comments as thread starts here ->
print(len(thread_comments))
count = 0
for key in thread_comments:
    count += len(thread_comments[key])

print("comment count", count)

json_file=open("../data/splits/labelled/binary/qa-threads-suomi24.jsonl","wt")
# loop keys
for start in thread_starts:
    save = {}
    tmp = {}
    for i in range(len(thread_comments[start])):
        tmp["start"] = thread_starts[start]
        tmp[f"comment{i}"] = thread_comments[start][i]
    save["id"] = start
    save["thread"] = tmp
    line=json.dumps(save,ensure_ascii=False)
    print(line,file=json_file)
        


# if author is aloittaja then can delete comment as well -> maybe just one like that? or others because the username can be changed? so no reason to take that out hmm
# okay there was another aloittaja so I'll do that deletion


# some threads are qa threads where thread starts is prepared to answer questions and the comments have questions -> are these okay? how to remove if wanted?
