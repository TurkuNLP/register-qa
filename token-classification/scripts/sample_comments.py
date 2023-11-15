#!/usr/bin/env python3

import sys
import json
import gzip

from random import sample
from logging import error
from argparse import ArgumentParser


def argparser():
    ap = ArgumentParser()
    #ap.add_argument('comments', help='jsonl')
    ap.add_argument('predictions', help='tsv')
    ap.add_argument('label', help='e.g. "QA_NEW"')
    ap.add_argument('low', type=float, help='e.g. 0.0')
    ap.add_argument('high', type=float, help='e.g. 0.1')
    ap.add_argument('number', type=int, help='e.g. 10')
    return ap

#mkdir samples/sample-parsebank
#for l in 0.5 0.6 0.7 0.8 0.9; do h=$(python3 -c 'print('$l'+0.1)'); python3 sample_comments.py ../qa-labelled/sorted/sorted_qa_binary_parsebank.tsv.gz QA_NEW $l $h 20 > ../samples/sample-parsebank/comments_${l}-${h}.jsonl; done


def main(argv):
    args = argparser().parse_args(argv[1:])

    ids = []
    texts = []
    with gzip.open(args.predictions, "rt") as f:
        # this header thing doesn't quite work
        header = next(f).rstrip('\n')
        try:
            index = header.split('\t').index(args.label)
        except ValueError:
            error(f'label "{args.label}" not found in header "{header}"')
            return 1

        for ln, line in enumerate(f, start=2):
            fields = line.rstrip('\n').split('\t')
            id_, text, value = fields[0], fields[2], fields[4] # oh id_, the underscore is just for using that keyword as variable :D
            value = float(value)
            if args.low <= value <= args.high:
                ids.append(id_)
                texts.append(text)

    lista = []
    for id_, text in zip(ids, texts):
        temp = {}
        # put the newlines back the way they were before tsv
        text = text.replace("\\n", "\n")
        temp["id"] = id_
        temp["text"] = text
        lista.append(temp)

    if len(ids) < args.number:
        #error(f'cannot sample {args.number} from range: only {len(ids)} found')
        error(f'cannot sample {args.number}: only {len(ids)} found')
        return 1
    id_sample = sample(ids, args.number)

    # with open(args.comments) as f:
    #     for line in f:
    #         data = json.loads(line)
    #         if data['id'] in id_sample:
    #             print(line, end='')

    sample_list = [one for one in lista if one["id"] in id_sample]

    for one in sample_list:
        line=json.dumps(one,ensure_ascii=False)
        print(line, end='\n')



if __name__ == '__main__':
    sys.exit(main(sys.argv))
