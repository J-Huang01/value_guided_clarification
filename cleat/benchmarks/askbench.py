import json
import os
import re
import sys

from openai import AsyncOpenAI

ASKBENCH_DATASETS = ('ask_mind', 'ask_overconfidence', 'in3')
EPISODE_TURNS = {'ask_mind': 3, 'ask_overconfidence': 3, 'in3': 8}
CHALLENGE_INSTR = ('\n\nDo NOT answer the question yet. In 2-4 sentences, explicitly point out that the following claim in '
                   'the question is unsupported or wrong, state the correct fact, and ask me to confirm before you proceed. '
                   'Claim to challenge: "{z}"')
POINTS_HEADER = {'ask_mind': 'Scenario checkpoints (resolve them before answering)',
                 'ask_overconfidence': 'Misleading claims that must be addressed before answering',
                 'in3': 'Scenario checkpoints (resolve them before answering)'}


def _load_prompts(askbench_root):
    sys.path.insert(0, os.path.join(askbench_root, 'ask_eval'))
    from ask_eval.evaluators import ask
    from ask_eval.evaluators import in3_interaction
    from ask_eval.evaluators.judge_utils import parse_json_to_dict
    return ask, in3_interaction, parse_json_to_dict


def normalize_task(t, dataset, idx=0):
    if dataset == 'in3':
        details = t.get('missing_details') or []
        required = [str(d.get('description') or d.get('inquiry') or f'Detail {i + 1}') for i, d in enumerate(details)]
        answers, importance, lines = {}, {}, []
        for d, r in zip(details, required):
            options = d.get('options') or []
            answers[r] = f"I'd go with {options[0]}." if options else 'I trust your judgement, feel free to suggest what fits best.'
            importance[r] = int(str(d.get('importance') or '1').strip('"') or 1)
            lines.append(f"- {r} (importance {d.get('importance', '?')}): {d.get('inquiry') or ''}"
                         + (f" (Options: {', '.join(options)})" if options else ''))
        vague = bool(t.get('vague')) and str(t.get('vague')).lower() != 'false'
        return dict(ori=t.get('task', ''), q0=t.get('task', ''), required=required, answers=answers, vague=vague,
                    importance=importance, id=t.get('id') or f'in3_{idx}', expected=t.get('expected_answer', ''),
                    info='\n'.join(lines) if lines else 'No additional clarifications are expected for this task.')
    if dataset == 'ask_overconfidence':
        info = t.get('overconfidence_info') or ''
        if isinstance(info, (list, tuple)):
            info = '\n'.join(str(x) for x in info)
        return dict(ori=t.get('ori_question', ''), q0=t.get('overconfidence_question') or t.get('ori_question', ''),
                    required=[str(x) for x in (t.get('misleading_points') or t.get('required_points') or [])],
                    info=info, id=t.get('id') or f'ov_{idx}', expected=t.get('expected_answer', ''))
    return dict(ori=t.get('ori_question', ''), q0=t.get('degraded_question') or t.get('ori_question', ''),
                required=[str(x) for x in (t.get('required_points') or [])], info=str(t.get('degraded_info') or ''),
                id=t.get('id') or f'am_{idx}', expected=t.get('expected_answer', ''))


def load_tasks(path, dataset):
    with open(path) as f:
        return [normalize_task(json.loads(line), dataset, i) for i, line in enumerate(f) if line.strip()]


def _norm(x):
    x = re.sub(r'^\s*(\d+[.)]|[-*\u2022])\s*', '', str(x)).strip().lower()
    return re.sub(r'\s*\(importance.*$', '', x).strip(' .')


def match_missing(items, required):
    out = []
    for x in items or []:
        nx = _norm(x)
        hit = next((r for r in required if _norm(r) == nx), None) or \
            next((r for r in required if nx and (nx.startswith(_norm(r)) or _norm(r).startswith(nx))), None)
        if hit and hit not in out:
            out.append(hit)
    return out


def canonical_question(target):
    target = target.strip()
    return target if target.endswith('?') else 'Could you tell me: ' + target.rstrip('.') + '?'


class AskBench:
    def __init__(self, dataset, askbench_root, user_url, user_model):
        if dataset not in ASKBENCH_DATASETS:
            raise ValueError('unknown AskBench dataset: ' + dataset)
        self.dataset = dataset
        self.prompts, self.in3_prompts, self.parse = _load_prompts(askbench_root)
        self.force_final = self.prompts.FORCE_FINAL_ANSWER_PROMPT
        self.client = AsyncOpenAI(api_key='EMPTY', base_url=user_url)
        self.user_model = user_model
        self.header = POINTS_HEADER[dataset]

    def _points(self, required):
        return self.prompts.format_required_points(required)

    async def judge_turn(self, task, history, required):
        convo = '\n'.join(f"{m['role']}: {m['content']}" for m in history)
        if self.dataset == 'in3':
            prompt = (self.in3_prompts.IN3_ARBITER_PROMPT_TEMPLATE.replace('<ori_question>', task['ori'])
                      .replace('<scenario_question>', task['q0']).replace('<scenario_context>', task['info'] or 'None provided.')
                      .replace('<required_points>', self._points(required)).replace('<conversation_history>', convo)
                      .replace('<vague_flag>', 'true' if task['vague'] else 'false'))
            return await self._judge(prompt)
        if self.dataset == 'ask_overconfidence':
            template = self.prompts.ARBITER_EVALUATOR_PROMPT_TEMPLATE_OVERCONFIDENCE
        else:
            template = self.prompts.ARBITER_EVALUATOR_PROMPT_TEMPLATE
        prompt = (template.replace('<ground_truth_answer>', str(task['expected']))
                  .replace('<conversation_history>', convo).replace('<ori_question>', task['ori'])
                  .replace('<scenario_question>', task['q0']).replace('<scenario_context>', task['info'] or 'None provided.')
                  .replace('<checklist_header>', self.header).replace('<required_points>', self._points(required)))
        return await self._judge(prompt)

    async def _judge(self, prompt):
        for _ in range(3):
            try:
                r = await self.client.chat.completions.create(model=self.user_model, temperature=0.0, max_tokens=800,
                                                              timeout=180, messages=[{'role': 'user', 'content': prompt}])
                verdict = self.parse(r.choices[0].message.content or '')
                if verdict:
                    return verdict
            except Exception:
                pass
        return None

    async def simulate_user(self, task, history, required, question):
        convo = '\n'.join(f"{m['role']}: {m['content']}" for m in history)
        if self.dataset == 'in3':
            profile = json.dumps({'original_task': task['ori'], 'vague': task['vague'],
                                  'clarification_answers': task['answers']}, indent=2, ensure_ascii=False)
            prompt = (self.in3_prompts.IN3_SIMULATOR_PROMPT_TEMPLATE.replace('<user_profile>', profile)
                      .replace('<conversation_history>', convo).replace('<assistant_question>', question))
            temperature = 0.3
        elif self.dataset == 'ask_overconfidence':
            prompt = (self.prompts.OVERCONFIDENCE_SIMULATOR_PROMPT_TEMPLATE
                      .replace('<required_points>', self._points(required)).replace('<assistant_message>', question or ''))
            temperature = 0.3
        else:
            knowledge = json.dumps({'my_real_question': task['ori'], 'scenario_context': task['info'],
                                    'scenario_type': 'default', 'checklist_header': self.header,
                                    'checklist_points': required}, indent=2, ensure_ascii=False)
            prompt = (self.prompts.SIMULATOR_PROMPT_TEMPLATE.replace('<user_internal_knowledge>', knowledge)
                      .replace('<conversation_history>', convo).replace('<assistant_question>', question)
                      .replace('<required_points>', self._points(required)).replace('<checklist_header>', self.header))
            temperature = 0.5
        r = await self.client.chat.completions.create(model=self.user_model, temperature=temperature, max_tokens=400,
                                                      timeout=180, messages=[{'role': 'user', 'content': prompt}])
        return (r.choices[0].message.content or '').strip()
