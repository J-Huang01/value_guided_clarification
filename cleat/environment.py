import asyncio
import copy
import hashlib
import json
import os

from benchmarks import JudgeFailure
from benchmarks import askbench as ab
from benchmarks import tau2 as t2
from benchmarks import usergym as ug

JUDGE_RETRIES = 3



def fingerprint(task, dataset):
    if dataset == 'tau2':
        text = f"{task['env_name']}|{task['gold']}"
    else:
        text = task['q0'] if dataset in ab.ASKBENCH_DATASETS else ug.flatten(task['messages'])
    return hashlib.sha256(' '.join(text.split()).encode()).hexdigest()


def identity(task, dataset):
    if dataset in ab.ASKBENCH_DATASETS:
        return fingerprint(task, dataset)
    return hashlib.sha256(f"{task['env_name']}|{task['gold']}".encode()).hexdigest()


def _unique(tasks, excluded, dataset):
    seen, out = set(), []
    for t in tasks:
        if fingerprint(t, dataset) in excluded:
            continue
        key = identity(t, dataset)
        if key not in seen:
            out.append(dict(t, uid=key))
            seen.add(key)
    return out


def load_splits(args):
    dataset = args.dataset
    if dataset in ab.ASKBENCH_DATASETS:
        data_dir = args.data_dir or os.path.join('data', dataset)
        train = ab.load_tasks(os.path.join(data_dir, 'train.jsonl'), dataset)
        dev = ab.load_tasks(os.path.join(data_dir, 'dev.jsonl'), dataset)
        test = ab.load_tasks(os.path.join(data_dir, 'test.jsonl'), dataset)
    elif dataset == 'tau2':
        pool, test = [], []
        for domain in args.tau2_domains:
            pool.extend(t2.load_tasks(domain, 'train'))
            test.extend(t2.load_tasks(domain, 'test'))
        train, dev = [], []
        for t in pool:
            (dev if int(fingerprint(t, dataset)[:8], 16) % 10 == 0 else train).append(t)
    else:
        pool, test = [], []
        for name in ug.TRAIN_ENVS:
            pool.extend(ug.load_tasks(args.userrl_root, name, 'train'))
        for name in ug.TRAIN_ENVS + ug.HELD_OUT_ENVS:
            test.extend(ug.load_tasks(args.userrl_root, name, 'test'))
        train, dev = [], []
        for t in pool:
            (dev if int(fingerprint(t, dataset)[:8], 16) % 10 == 0 else train).append(t)
    test_keys = {fingerprint(t, dataset) for t in test}
    dev_keys = {fingerprint(t, dataset) for t in dev}
    splits = (_unique(train, dev_keys | test_keys, dataset), _unique(dev, test_keys, dataset), _unique(test, set(), dataset))
    if not all(splits):
        raise ValueError('empty split after removing cross-split duplicates')
    return splits


class Bench:
    def __init__(self, args, actor):
        self.dataset, self.actor = args.dataset, actor
        self.ask = args.dataset in ab.ASKBENCH_DATASETS
        self.episodic = args.dataset == 'tau2'
        if self.episodic:
            self.limit = t2.MAX_STEPS // 2
            self.env = t2.Tau2(actor, args.user_url, args.user_model)
        elif self.ask:
            self.limit = ab.EPISODE_TURNS[args.dataset]
            self.env = ab.AskBench(args.dataset, args.askbench_root, args.user_url, args.user_model)
        else:
            self.limit = ug.EPISODE_STEPS
            self.env = ug.UserGym(args.userrl_root, args.user_url, args.user_model, actor.counts)

    def initial(self, task):
        env = None if self.ask else self.env.build_env(task, self.limit)
        messages = [{'role': 'user', 'content': task['q0']}] if self.ask else copy.deepcopy(task['messages'])
        return dict(task=task, env=env, messages=messages, turn=0, done=False, asked=[], queries=0,
                    native_queries=0, reward=0.0, correct=None, missing=list(task.get('required', [])), turns=[],
                    step_rewards=[])

    def fork(self, state):
        return copy.deepcopy(state)

    async def labels(self, state):
        if state['done'] or (self.ask and state['turn'] == self.limit - 1):
            return []
        return await self.actor.propose(state['messages'], state['asked'])

    async def _say(self, state, label, seed, record_loss, greedy, temperature):
        records = []
        if label is None:
            msgs = copy.deepcopy(state['messages'])
            if self.ask and state['turn'] == self.limit - 1:
                msgs[-1]['content'] += '\n' + self.env.force_final
            text, rec = await self.actor.generate(msgs, seed, tools=None if self.ask else [self.env.schema],
                                                  record_loss=record_loss, greedy=greedy, temperature=temperature)
            if rec:
                records.append(rec)
        elif self.dataset == 'ask_overconfidence':
            msgs = copy.deepcopy(state['messages'])
            msgs[-1]['content'] += ab.CHALLENGE_INSTR.format(z=label)
            text, _ = await self.actor.generate(msgs, seed, frozen=True, record_loss=False, max_new=220)
        else:
            text = ab.canonical_question(label)
        if label is not None:
            state['asked'].append(label)
            state['queries'] += 1
        return text, records

    async def step(self, state, label, seed, record_loss, greedy=False, temperature=1.0):
        text, records = await self._say(state, label, seed, record_loss, greedy, temperature)
        self.actor.counts['environment_steps'] += 1
        state['turn'] += 1
        if self.ask:
            await self._askbench_step(state, label, text)
        else:
            await self._usergym_step(state, label, text)
        state['turns'].append(dict(turn=state['turn'], semantic_query=label, text=text))
        return records

    async def _usergym_step(self, state, label, text):
        choice, content, call = ('action', text, None) if label is not None else ug.parse_tool_call(text)
        if label is None:
            state['native_queries'] += int(ug.is_question(choice, content))
        obs, step_reward, terminated, truncated, _ = await self.env.env_step(state['env'], f'[{choice}] {content}')
        state['step_rewards'].append(float(step_reward or 0.0))
        feedback = str(obs.get('feedback', ''))
        if call:
            call_id = f"call_{state['turn']}"
            arguments = json.dumps(dict(choice=choice, content=content))
            state['messages'].append(dict(role='assistant', content=text.split('<tool_call>')[0].strip(),
                                          tool_calls=[dict(id=call_id, type='function',
                                                           function=dict(name='interact_with_env', arguments=arguments))]))
            state['messages'].append(dict(role='tool', tool_call_id=call_id, content=feedback))
        else:
            state['messages'] += [dict(role='assistant', content=text), dict(role='user', content=feedback)]
        env = state['env']
        state['reward'] = float(env.total_reward)
        state['done'] = bool(terminated or truncated or state['turn'] >= self.limit)
        state['coverage'] = float(env._calculate_elicitation_ratio()) if hasattr(env, '_calculate_elicitation_ratio') else None

    async def _askbench_step(self, state, label, text):
        task = state['task']
        state['messages'].append(dict(role='assistant', content=text))
        self.actor.counts['simulator_calls'] += 1
        verdict = None
        for attempt in range(1 + JUDGE_RETRIES):
            verdict = await self.env.judge_turn(task, state['messages'], task['required'])
            if verdict is not None and isinstance(verdict.get('is_final_answer'), bool) \
                    and isinstance(verdict.get('missing_required_points'), list):
                break
            if attempt:
                await asyncio.sleep(2.0 * attempt)
        if verdict is None or not isinstance(verdict.get('is_final_answer'), bool) \
                or not isinstance(verdict.get('missing_required_points'), list):
            raise JudgeFailure('judge returned no valid verdict')
        state['missing'] = ab.match_missing(verdict['missing_required_points'], task['required'])
        state['coverage'] = float(not state['missing'])
        required = task.get('required') or []
        state['info'] = (len(required) - len(state['missing'])) / len(required) if required else 0.0
        if verdict['is_final_answer']:
            correct = verdict.get('is_correct')
            if isinstance(correct, str) and correct.strip().lower() in ('true', 'false'):
                correct = correct.strip().lower() == 'true'
            state['correct'] = bool(correct) if isinstance(correct, bool) else False
            state['reward'] = state['info'] + float(state['correct'])
            state['done'] = True
            return
        state['native_queries'] += int(label is None)
        if state['turn'] >= self.limit:
            state['done'] = True
            return
        self.actor.counts['simulator_calls'] += 1
        reply = await self.env.simulate_user(task, state['messages'], task['required'], text)
        if not reply:
            raise RuntimeError('user simulator returned an empty reply')
        state['messages'].append(dict(role='user', content=reply))

    def result(self, state):
        env_name = state['task'].get('env_name', self.dataset)
        steps = state.get('step_rewards') or [0.0]
        pass1 = max(steps) if 'travel' in env_name else sum(steps)
        extra = {}
        if self.dataset == 'in3':
            task = state['task']
            weights = task.get('importance') or {}
            total = sum(weights.get(r, 1) for r in task['required'])
            resolved = sum(weights.get(r, 1) for r in task['required'] if r not in state['missing'])
            extra = dict(vague=task['vague'], asked=bool(state['queries'] + state['native_queries']),
                         recover=resolved / total if (task['vague'] and total) else None)
        return dict(uid=state['task']['uid'], env_name=env_name, pass1=pass1 if not self.ask else None, **extra,
                    reward=state['reward'], correct=state['correct'],
                    coverage=state.get('coverage', 0.0), info=state.get('info', 0.0), queries=state['queries'],
                    native_queries=state['native_queries'], turns=state['turn'], trace=state['turns'])
