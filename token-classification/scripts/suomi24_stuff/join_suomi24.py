import sys
import json

# first need to get suomi24 ids from the not!! sorted qa file, then join those ids with the original files .... will take forever so faster probably to just do the register thing again and get the other fields :D

def get_tsv(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")
        lines[i] = lines[i].split("\t")
    ids = [row[0] for row in lines] 

    # maybe probabilities taken as well?
    not_qa = [row[3] for row in lines] 
    qa = [row[4] for row in lines] 

    return ids, not_qa, qa


def compare_ids():
    qa_file = "../data/splits/labelled/binary/qa_binary_suomi24.tsv"
    qa_ids, not_qa_probs, qa_probs = get_tsv(qa_file)

    # here get the file name(s)

    json_file = sys.argv[1]

    found = 0
    with open(json_file, 'r') as f:
        #lines = f.readlines() 
        # file is too big, have to do this little bit at a time ehh, then cannot get 
        # have to do the batching or something to make go faster
        batch = [f.readline() for _ in range(0, 500)]
        batch = [line for line in batch if len(line) > 0]

        while batch:

            all_lines = [json.loads(line) for line in batch]
            ids = [json.loads(line)["id"] for line in batch]
            #texts = [json.loads(line)["text"] for line in lines]
            #meta = [json.loads(line)["meta"]["source_meta"] for line in lines]
            #print(meta[:5])


            joined = []
            if found == 0:
                for i in range(len(qa_ids)):
                    for j in range(len(ids)):
                        if qa_ids[i] == ids[j]:
                            joined.append(all_lines[j])
                            break
            else:
                for i in range(len(qa_ids[found:])):
                    for j in range(len(ids)):
                        if qa_ids[i+found] == ids[j]:
                            joined.append(all_lines[j])
                            break
                        if j == len(ids) -1:
                            found = found + i
                            break

            batch = [f.readline() for _ in range(0, 500)]
            batch = [line for line in batch if len(line) > 0]
    print(found)
    return joined


def main():
    joined = compare_ids()


    # save to jsonl again
    with open("joined_qa_suomi24.jsonl", "w") as outfile:
        for one in joined:
            outfile.write(one)
    


if __name__ == "__main__":
    main()