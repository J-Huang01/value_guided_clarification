import argparse
import asyncio
import copy
import json
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from actor import Actor
from environment import Bench, JudgeFailure, load_splits
from math_core import advantages, effective_sample_size, guidance, sample_arms, value_rule


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'object of type {o.__class__.__name__} is not JSON serializable')


def dumps(data, **kw):
    return json.dumps(data, ensure_ascii=False, allow_nan=False, default=_jsonable, **kw)


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(dumps(data, indent=2))
    temp.replace(path)


def append(path, data):
    with path.open('a') as f:
        f.write(dumps(data) + '\n')


class Experiment:
    def __init__(self, args, fresh=True):
        self.a = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        if fresh and (self.out / 'config.json').exists() and not args.resume_from:
            raise FileExistsError('output directory already holds a run; choose a new one')
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        self.rng = np.random.default_rng(args.seed)
        self.train, self.dev, self.test = load_splits(args)
        self.rng.shuffle(self.train)
        self.dev.sort(key=lambda t: t['uid'])
        self.dev = self.dev[:args.eval_tasks]
        self.cursor = 0
        self.step = 0
        self.start = time.time()
        self.judge_failures = 0
        self.actor = Actor(args)
        self.bench = Bench(args, self.actor)
        self.train_counts = {k: 0 for k in self.actor.counts}
        if fresh:
            write_json(self.out / ('config_resume.json' if args.resume_from else 'config.json'), vars(args))
            write_json(self.out / 'splits.json', {name: [t['uid'] for t in tasks] for name, tasks in
                                                  [('train', self.train), ('dev', self.dev), ('test', self.test)]})

    async def policy_turn(self, state, rng, beta=0.0, greedy=False, temperature=1.0, max_queries=None):
        use_menu = beta and not state.get('stopped') and (max_queries is None or state['queries'] < max_queries)
        labels = await self.bench.labels(state) if use_menu else []
        choice = len(labels)
        if labels:
            choice = value_rule(self.actor.values(state['messages'], labels, self.bench.limit - state['turn']))
            if choice == len(labels):
                state['stopped'] = True
        label = labels[choice] if choice < len(labels) else None
        await self.bench.step(state, label, int(rng.integers(2 ** 31)), False, greedy=greedy, temperature=temperature)

    async def finish(self, root, labels, arm, seed, record_loss=True):
        state = self.bench.fork(root)
        rng = np.random.default_rng(seed)
        records = await self.bench.step(state, labels[arm] if arm < len(labels) else None,
                                        int(rng.integers(2 ** 31)), record_loss)
        after_first = copy.deepcopy(state['messages'])
        while not state['done']:
            records += await self.bench.step(state, None, int(rng.integers(2 ** 31)), record_loss)
        return dict(result=self.bench.result(state), records=records, after_first=after_first)

    async def collect_group(self, task, seed, warmup=False):
        rng = np.random.default_rng(seed)
        root = self.bench.initial(task)
        for _ in range(int(rng.integers(self.a.prefix_max + 1))):
            if root['done']:
                break
            before = self.bench.fork(root)
            await self.policy_turn(root, rng)
            if root['done']:
                root = before
                break
        if root['done']:
            return dict(terminal_prefix=True, uid=task['uid'], result=self.bench.result(root))
        labels = await self.bench.labels(root)
        pi, menu = await self.actor.score_menu(root['messages'], labels, True)
        values = self.actor.values(root['messages'], labels, self.bench.limit - root['turn'])
        beta = 0.0 if warmup or self.a.variant == 'no_guidance' else self.a.beta
        shaped = guidance(values, self.a.guide_scale) if beta else values
        arms, rho, bar, mu = sample_arms(pi, shaped, beta, self.a.epsilon, rng)
        keys, jobs = [], []
        for arm, replicas in arms.items():
            for _ in range(replicas):
                keys.append(arm)
                jobs.append(self.finish(root, labels, arm, int(rng.integers(2 ** 31))))
        trajectories = await asyncio.gather(*jobs)
        by_arm = {arm: [] for arm in arms}
        for arm, tr in zip(keys, trajectories):
            by_arm[arm].append(tr)
        rewards = {arm: [t['result']['reward'] for t in trs] for arm, trs in by_arm.items()}
        return dict(uid=task['uid'], seed=seed, messages=root['messages'], labels=labels, menu=menu,
                    pi=pi.tolist(), rho=rho.tolist(), bar=bar.tolist(), mu=mu.tolist(), predictions=values.tolist(),
                    rewards=rewards, trajectories=by_arm, budget=self.bench.limit - root['turn'],
                    actor_version=self.actor.version, terminal_prefix=False)

    def records(self, groups):
        records, weights = [], []
        for g in groups:
            adv = advantages(g['rewards'], len(g['labels']))
            for arm, trajs in g['trajectories'].items():
                weight = g['pi'][arm] / g['rho'][arm]
                weights.append(weight)
                records.append((dict(g['menu'], action=arm), float(adv[arm].mean()), weight))
                for a, tr in zip(adv[arm], trajs):
                    records.extend((rec, float(a), weight / len(trajs)) for rec in tr['records'])
        return records, weights

    def save(self, label):
        path = self.out / 'checkpoints' / label
        path.mkdir(parents=True, exist_ok=True)
        self.actor.lm.save_pretrained(path / 'adapter')
        torch.save(dict(head=self.actor.head.state_dict(), actor_optimizer=self.actor.opt.state_dict(),
                        controller_optimizer=self.actor.hopt.state_dict(), version=self.actor.version,
                        numpy_rng=self.rng.bit_generator.state, step=self.step, cursor=self.cursor,
                        counts=self.actor.counts, train_counts=self.train_counts), path / 'training.pt')

    def load_checkpoint(self, source):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        path = Path(source)
        saved = torch.load(path / 'training.pt', map_location='cpu', weights_only=False)
        set_peft_model_state_dict(self.actor.lm, load_file(str(path / 'adapter' / 'adapter_model.safetensors')))
        self.actor.head.load_state_dict(saved['head'])
        self.actor.opt.load_state_dict(saved['actor_optimizer'])
        self.actor.hopt.load_state_dict(saved['controller_optimizer'])
        self.actor.version = saved['version']
        self.cursor = saved['cursor']
        self.rng.bit_generator.state = saved['numpy_rng']
        self.actor.counts = saved['counts']
        self.train_counts = saved['train_counts']
        self.step = int(saved['step'])
        return self.step

    async def batch(self, warmup=False, refit=False):
        before = dict(self.actor.counts)
        picks = []
        for _ in range(self.a.batch_tasks):
            picks.append((self.train[self.cursor % len(self.train)], int(self.rng.integers(2 ** 31))))
            self.cursor += 1
        limit = asyncio.Semaphore(self.a.collect_concurrency or self.a.batch_tasks)

        async def collect(task, seed):
            async with limit:
                try:
                    return await self.collect_group(task, seed, warmup)
                except JudgeFailure as error:
                    self.judge_failures += 1
                    append(self.out / 'dropped_groups.jsonl', dict(step=self.step, uid=task['uid'], error=repr(error)))
                    return None
        started = time.time()
        collected = await asyncio.gather(*[collect(task, seed) for task, seed in picks])
        collect_s = time.time() - started
        dropped = sum(g is None for g in collected)
        if dropped >= max(1, len(picks) // 2):
            raise RuntimeError(f'judge failed on {dropped}/{len(picks)} groups of one batch')
        groups = [g for g in collected if g is not None and not g['terminal_prefix']]
        for g in groups:
            audit = {k: v for k, v in g.items() if k not in ('trajectories', 'menu')}
            audit['trajectories'] = {i: [t['result'] for t in ts] for i, ts in g['trajectories'].items()}
            append(self.out / 'groups.jsonl', dict(audit, step=self.step, warmup=warmup))
        for k in self.train_counts:
            self.train_counts[k] += self.actor.counts[k] - before[k]
        metrics = dict(collect_s=round(collect_s, 1))
        if groups:
            records, weights = self.records(groups)
            if not warmup and not refit:
                metrics.update(self.actor.update_actor(records, len(groups)))
                await self.actor.sync_weights()
            if warmup or self.a.variant != 'frozen_controller':
                metrics.update(self.actor.update_head(groups))
            rewards = [v for g in groups for vs in g['rewards'].values() for v in vs]
            query_values = [np.mean(rs) - np.mean(g['rewards'][len(g['labels'])])
                            for g in groups for i, rs in g['rewards'].items() if i < len(g['labels'])]
            metrics.update(rollout_reward=float(np.mean(rewards)), importance_ess=effective_sample_size(weights),
                           mean_query_value=float(np.mean(query_values)) if query_values else None)
        row = dict(step=self.step, warmup=warmup, refit=refit, elapsed_s=time.time() - self.start, groups=len(groups),
                   dropped_groups=dropped, actor_version=self.actor.version, train_counts=dict(self.train_counts),
                   total_counts=dict(self.actor.counts), **metrics)
        append(self.out / 'train.jsonl', row)
        print(dumps(row), flush=True)

    async def evaluate(self):
        limit = asyncio.Semaphore(self.a.eval_concurrency)

        async def one(beta, i, task):
            async with limit:
                state = self.bench.initial(task)
                rng = np.random.default_rng(900000 + i)
                try:
                    while not state['done']:
                        await self.policy_turn(state, rng, beta=beta, greedy=True, max_queries=self.a.max_queries)
                except JudgeFailure as error:
                    self.judge_failures += 1
                    append(self.out / 'dropped_groups.jsonl', dict(step=self.step, uid=task['uid'], error=repr(error)))
                    return
                append(self.out / 'eval.jsonl', dict(self.bench.result(state), step=self.step, beta=beta,
                                                     split='dev', actor_version=self.actor.version))
        for beta in (0.0, 1.0):
            await asyncio.gather(*[one(beta, i, task) for i, task in enumerate(self.dev)])

    async def run(self):
        try:
            start = 1
            if self.a.resume_from:
                start = self.load_checkpoint(self.a.resume_from) + 1
                await self.actor.sync_weights()
            else:
                for k in range(self.a.warmup_steps):
                    self.step = k - self.a.warmup_steps
                    await self.batch(warmup=True)
                self.step = 0
                self.save('warmup')
                await self.evaluate()
            for step in range(start, self.a.steps + 1):
                self.step = step
                await self.batch()
                if step % self.a.eval_every == 0 or step == self.a.steps:
                    self.save(f'step-{step:04d}')
                    await self.evaluate()
            self.step += 1
            await self.batch(refit=True)
            self.save('final-refit')
            write_json(self.out / 'complete.json', dict(steps=self.a.steps, elapsed_s=time.time() - self.start,
                                                        total_counts=self.actor.counts, judge_failures=self.judge_failures))
        except BaseException as error:
            write_json(self.out / 'failed.json', dict(step=self.step, error=repr(error), traceback=traceback.format_exc()))
            raise


def add_common_args(p):
    p.add_argument('--dataset', choices=['ask_mind', 'ask_overconfidence', 'usergym'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--served-name', default=None)
    p.add_argument('--agent-url', default=os.environ.get('AGENT_URL', 'http://localhost:8000/v1'))
    p.add_argument('--user-url', default=os.environ.get('USER_URL', 'http://localhost:8100/v1'))
    p.add_argument('--user-model', default=os.environ.get('USER_MODEL', 'Qwen3-30B-A3B-Instruct-2507'))
    p.add_argument('--askbench-root', default=os.environ.get('ASKBENCH_ROOT', 'third_party/askbench'))
    p.add_argument('--userrl-root', default=os.environ.get('USERRL_ROOT', 'third_party/UserRL'))
    p.add_argument('--data-dir', default=None)
    p.add_argument('--topm', type=int, default=None)
    p.add_argument('--lora-rank', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--controller-lr', type=float, default=3e-4)
    p.add_argument('--controller-epochs', type=int, default=4)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--kl-coef', type=float, default=0.01)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--context-length', type=int, default=8192)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--logprob-concurrency', type=int, default=6)
    p.add_argument('--eval-tasks', type=int, default=64)
    p.add_argument('--eval-concurrency', type=int, default=64)
    p.add_argument('--max-queries', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    return p


def parse_args(argv=None):
    p = add_common_args(argparse.ArgumentParser())
    p.add_argument('--out', required=True)
    p.add_argument('--variant', choices=['online', 'no_guidance', 'frozen_controller'], default='online')
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--warmup-steps', type=int, default=4)
    p.add_argument('--batch-tasks', type=int, default=16)
    p.add_argument('--collect-concurrency', type=int, default=32)
    p.add_argument('--prefix-max', type=int, default=1)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--epsilon', type=float, default=0.2)
    p.add_argument('--guide-scale', type=float, default=18.0)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--resume-from', default=None)
    a = p.parse_args(argv)
    if a.max_queries is None:
        a.max_queries = 4 if a.dataset == 'usergym' else 1
    if a.topm is None:
        a.topm = 6
    if a.steps < 1 or a.warmup_steps < 1 or not 0 < a.epsilon <= 1:
        p.error('need at least one training step, one warm-up step and epsilon in (0, 1]')
    return a


if __name__ == '__main__':
    asyncio.run(Experiment(parse_args()).run())
