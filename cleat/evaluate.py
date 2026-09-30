import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from environment import JudgeFailure
from train import Experiment, add_common_args, append, dumps


def parse_args(argv=None):
    p = add_common_args(argparse.ArgumentParser())
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--temperature', type=float, default=None)
    p.add_argument('--arms', nargs='+', choices=['controller', 'agent_only'], default=['controller', 'agent_only'])
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args(argv)
    a.resume_from = None
    usergym = a.dataset == 'usergym'
    if a.temperature is None:
        a.temperature = 0.0 if usergym else 0.7
    if a.max_queries is None:
        a.max_queries = 4 if usergym else 1
    if a.topm is None:
        a.topm = 3
    return a


async def main(args):
    exp = Experiment(args, fresh=False)
    ckpt = Path(args.checkpoint)
    saved = torch.load(ckpt / 'training.pt', map_location='cpu', weights_only=False)
    set_peft_model_state_dict(exp.actor.lm, load_file(str(ckpt / 'adapter' / 'adapter_model.safetensors')))
    exp.actor.head.load_state_dict(saved['head'])
    exp.actor.version = int(saved['version'])
    await exp.actor.sync_weights()
    tasks = exp.test[:args.limit] if args.limit else exp.test
    out = Path(args.out)
    limit = asyncio.Semaphore(args.eval_concurrency)
    summary = {}

    async def one(arm, seed, i, task):
        async with limit:
            state = exp.bench.initial(task)
            rng = np.random.default_rng(seed * 1000003 + i)
            try:
                while not state['done']:
                    await exp.policy_turn(state, rng, beta=1.0 if arm == 'controller' else 0.0,
                                          temperature=args.temperature, max_queries=args.max_queries)
            except JudgeFailure:
                return None
            result = dict(exp.bench.result(state), arm=arm, seed=seed)
            append(out / 'test_results.jsonl', result)
            return result

    for arm in args.arms:
        for seed in args.seeds:
            results = [r for r in await asyncio.gather(*[one(arm, seed, i, t) for i, t in enumerate(tasks)]) if r]
            row = dict(arm=arm, seed=seed, n=len(results),
                       reward=float(np.mean([r['reward'] for r in results])),
                       coverage=float(np.mean([r['coverage'] or 0.0 for r in results])),
                       queries=float(np.mean([r['queries'] for r in results])))
            if exp.bench.ask:
                row['accuracy'] = float(np.mean([bool(r['correct']) for r in results]))
            summary.setdefault(arm, []).append(row)
            print(dumps(row), flush=True)
    report = {}
    for arm, rows in summary.items():
        report[arm] = {k: dict(mean=float(np.mean([r[k] for r in rows])), std=float(np.std([r[k] for r in rows], ddof=1))
                               if len(rows) > 1 else 0.0) for k in rows[0] if k not in ('arm', 'seed', 'n')}
    (out / 'summary.json').write_text(json.dumps(dict(per_seed=summary, aggregate=report), indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    arguments = parse_args()
    Path(arguments.out).mkdir(parents=True, exist_ok=True)
    asyncio.run(main(arguments))
