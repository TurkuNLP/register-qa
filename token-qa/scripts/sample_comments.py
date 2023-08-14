#!/usr/bin/env python3

import sys
import json

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

#mkdir sample-01-obscene
#for l in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do h=$(python -c 'print('$l'+0.1)'); python3 sample_comments.py suomi24-2001-2020-1p-sample.jsonl s24predictions.tsv obscene $l $h 10 > sample-01-obscene/comments_${l}-${h}.jsonl; done


def main(argv):
    args = argparser().parse_args(argv[1:])

    ids = []
    with open(args.predictions) as f:
        header = next(f).rstrip('\n')
        try:
            index = header.split('\t').index(args.label)
        except ValueError:
            error(f'label "{args.label}" not found in header "{header}"')
            return 1

        for ln, line in enumerate(f, start=2):
            fields = line.rstrip('\n').split('\t')
            id_, value = fields[0], fields[index]
            value = float(value)
            if args.low <= value <= args.high:
                ids.append(id_)

    if len(ids) < args.number:
        error(f'cannot sample {args.number} from range: only {len(ids)} found')
        return 1
    id_sample = sample(ids, args.number)
    
    # with open(args.comments) as f:
    #     for line in f:
    #         data = json.loads(line)
    #         if data['id'] in id_sample:
    #             print(line, end='')

    #  TODO just print the line 


if __name__ == '__main__':
    sys.exit(main(sys.argv))
