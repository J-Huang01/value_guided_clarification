import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from environment import JudgeFailure
from train import QUESTION_BUDGET, Experiment, add_common_args, append, dumps


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
    if a.temperature is None:
        a.temperature = 0.0 if a.dataset in ('usergym', 'tau2') else 0.7
    if a.max_queries is None:
        a.max_queries = QUESTION_BUDGET.get(a.dataset, 1)
    if a.topm is None:
        a.topm = 3
    if a.context_length is None:
        a.context_length = 16384 if a.dataset == 'tau2' else 8192
    return a


async def main(args):
    exp = Experiment(args, fresh=False)
    ckpt = Path(args.checkpoint)
    saved = torch.load(ckpt / 'training.pt', map_location='cpu', weights_only=False)
    set_peft_model_state_dict(exp.actor.lm, load_file(str(ckpt / 'adapter' / 'adapter_model.safetensors')))
    exp.actor.head.load_state_dict(saved['head'])
    exp.actor.version = int(saved['version'])
    await exp.actor.sync_weights()
    tasks = exp.test
    if args.limit:
        per_env = {}
        for t in exp.test:
            per_env.setdefault(t.get('env_name', args.dataset), []).append(t)
        tasks = [t for group in per_env.values() for t in group[:args.limit]]
    out = Path(args.out)
    limit = asyncio.Semaphore(args.eval_concurrency)
    summary = {}

    async def one(arm, seed, i, task):
        async with limit:
            try:
                result = await exp.run_episode(task, seed * 1000003 + i, 1.0 if arm == 'controller' else 0.0,
                                               args.temperature == 0.0, args.temperature, args.max_queries)
            except JudgeFailure:
                return None
            result = dict(result, arm=arm, seed=seed)
            append(out / 'test_results.jsonl', result)
            return result

    for arm in args.arms:
        for seed in args.seeds:
            results = [r for r in await asyncio.gather(*[one(arm, seed, i, t) for i, t in enumerate(tasks)]) if r]
            row = dict(arm=arm, seed=seed, n=len(results),
                       reward=float(np.mean([r['reward'] for r in results])),
                       coverage=float(np.mean([r['coverage'] or 0.0 for r in results])),
                       queries=float(np.mean([r['queries'] for r in results])))
            if args.dataset == 'in3':
                recover = [r['recover'] for r in results if r.get('recover') is not None]
                row['recover'] = float(np.mean(recover)) if recover else 0.0
                row['judgment_accuracy'] = float(np.mean([r['asked'] == r['vague'] for r in results]))
            elif exp.bench.ask:
                row['accuracy'] = float(np.mean([bool(r['correct']) for r in results]))
            else:
                groups = {}
                for r in results:
                    groups.setdefault(r['env_name'], []).append(r['pass1'])
                    if 'travel' in r['env_name']:
                        groups.setdefault('travel', []).append(r['pass1'])
                for name in sorted(groups):
                    row[f'pass1/{name}'] = float(np.mean(groups[name]))
            summary.setdefault(arm, []).append(row)
            print(dumps(row), flush=True)
    report = {}
    for arm, rows in summary.items():
        keys = [k for k in rows[0] if k not in ('arm', 'seed', 'n')]
        report[arm] = {}
        for k in keys:
            values = [r[k] for r in rows if k in r]
            report[arm][k] = dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
    (out / 'summary.json').write_text(json.dumps(dict(per_seed=summary, aggregate=report), indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    arguments = parse_args()
    Path(arguments.out).mkdir(parents=True, exist_ok=True)
    asyncio.run(main(arguments))
