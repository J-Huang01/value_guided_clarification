import argparse
import json
import os
import random


def question(row):
    text = row.get('overconfidence_question') or row.get('degraded_question') or row.get('ori_question') or ''
    return ' '.join(str(text).split())


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path, rows):
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['ask_mind', 'ask_overconfidence'], required=True)
    p.add_argument('--pool', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--out-dir', default='data')
    p.add_argument('--train-size', type=int, default=2000)
    p.add_argument('--dev-size', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    test = read(a.test)
    held_out = {question(r) for r in test} | {' '.join(str(r.get('ori_question', '')).split()) for r in test}
    seen, pool = set(), []
    for row in read(a.pool):
        q = question(row)
        if q and q not in held_out and q not in seen and ' '.join(str(row.get('ori_question', '')).split()) not in held_out:
            seen.add(q)
            pool.append(row)
    random.Random(a.seed).shuffle(pool)
    if len(pool) < a.train_size + a.dev_size:
        raise SystemExit(f'pool has {len(pool)} usable tasks, fewer than train + dev')
    out = os.path.join(a.out_dir, a.dataset)
    os.makedirs(out, exist_ok=True)
    write(os.path.join(out, 'dev.jsonl'), pool[:a.dev_size])
    write(os.path.join(out, 'train.jsonl'), pool[a.dev_size:a.dev_size + a.train_size])
    write(os.path.join(out, 'test.jsonl'), test)
    print(f'{a.dataset}: train {a.train_size}, dev {a.dev_size}, test {len(test)} -> {out}')


if __name__ == '__main__':
    main()
