import asyncio
import hashlib
import json
import os
import re
import shutil
import uuid

import numpy as np
import requests
import torch
from openai import AsyncOpenAI
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def visible_text(messages):
    lines = []
    for m in messages:
        calls = json.dumps(m['tool_calls'], ensure_ascii=False) if m.get('tool_calls') else ''
        lines.append(f"{m['role']}: {m.get('content') or ''} " + calls)
    return '\n'.join(lines)


class ValueHead(nn.Module):
    def __init__(self, d, width=128):
        super().__init__()
        self.s = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, width), nn.Tanh())
        self.z = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, width), nn.Tanh())
        self.out = nn.Linear(width * 3 + 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, state, targets, budget):
        s = self.s(state.float()).expand(len(targets), -1)
        z = self.z(targets.float())
        b = torch.full((len(z), 1), budget / 16.0, device=z.device)
        return self.out(torch.cat([s, z, s * z, b], -1)).squeeze(-1)


LIST_PROMPT = ('List up to {words} specific missing pieces of information that would help complete the task in this '
               'conversation. Do not list information already provided. One target per line; only the list, or NONE if '
               'nothing is missing.\n\n')
CLAIM_PROMPT = ('The following question contains confident claims or reasoning that may be unsupported or misleading. '
                'List the up to {m} claims that you would most want to challenge before answering, one per line, most '
                'doubtful first. Output only the list.\n\nQuestion: ')
MENU_PROMPT = ('Choose the next high-level action. QUERY asks the user about the named target. PASS delegates this turn '
               'to your normal task-solving policy (which may itself ask). Copy exactly one line from this menu:\n')
NUMBER_WORDS = {3: 'three', 4: 'four', 5: 'five', 6: 'six', 8: 'eight'}
EMPTY_ITEMS = ('none', 'nothing', 'n/a', 'no missing information')


class Actor:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda:0'
        self.tok = AutoTokenizer.from_pretrained(args.model)
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                    attn_implementation='sdpa', device_map=self.device)
        lora = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
                          target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                          task_type='CAUSAL_LM')
        self.lm = get_peft_model(base, lora)
        self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.lm.enable_input_require_grads()
        self.lm.eval()
        self.opt = torch.optim.AdamW([p for p in self.lm.parameters() if p.requires_grad], lr=args.lr)
        self.head = ValueHead(base.config.hidden_size).to(self.device)
        self.hopt = torch.optim.AdamW(self.head.parameters(), lr=args.controller_lr)
        self.proposals = {}
        self.embeddings = {}
        self.counts = dict(agent_tokens=0, frozen_tokens=0, environment_steps=0, simulator_calls=0)
        self.version = 0
        self.client = AsyncOpenAI(api_key='EMPTY', base_url=args.agent_url, timeout=600.0, max_retries=3)
        self.base_name = args.served_name or args.model
        self.adapter_name = None
        self.session = uuid.uuid4().hex[:8]
        self.lora_dir = os.path.join(args.out, 'lora')
        self._logprob_limit = asyncio.Semaphore(args.logprob_concurrency)

    def prompt(self, messages, tools=None):
        kw = dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if tools:
            kw['tools'] = tools
        ids = self.tok.apply_chat_template(messages, **kw)
        return ids[-(self.args.context_length - self.args.max_new_tokens):]

    def token_lp(self, prefix, target):
        ids = torch.tensor([prefix + target], device=self.device)
        logits = self.lm(input_ids=ids, use_cache=False, logits_to_keep=len(target) + 1).logits
        logits = logits[:, -(len(target) + 1):-1, :].float()
        labels = ids[:, -len(target):]
        return -nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction='none')

    def menu_record(self, messages, labels):
        options = ['<QUERY> ' + z for z in labels] + ['<PASS>']
        msgs = [{'role': 'user', 'content': visible_text(messages) + '\n\n' + MENU_PROMPT + '\n'.join(options)}]
        return dict(kind='menu', prefix=self.prompt(msgs),
                    targets=[self.tok.encode(x, add_special_tokens=False) for x in options])

    def menu_lp(self, record):
        scores = torch.stack([self.token_lp(record['prefix'], t).mean() for t in record['targets']])
        return scores.log_softmax(0)

    async def score_menu(self, messages, labels, record_loss=True):
        record = self.menu_record(messages, labels)
        old = await self._menu_lp(record, self._policy_model())
        ref = await self._menu_lp(record, self.base_name) if record_loss else old
        record.update(old=old, ref=ref, version=self.version)
        return np.exp(old), record

    def _policy_model(self):
        return self.adapter_name or self.base_name

    async def _prompt_logprobs(self, model, ids, n_last):
        async with self._logprob_limit:
            r = await self.client.completions.create(model=model, prompt=ids, max_tokens=1, temperature=0.0,
                                                     extra_body={'prompt_logprobs': 0})
        values = []
        for entry in r.choices[0].prompt_logprobs[-n_last:]:
            if not entry:
                values.append(0.0)
                continue
            v = next(iter(entry.values()))
            values.append(float(v['logprob'] if isinstance(v, dict) else v.logprob))
        return values

    async def _menu_lp(self, record, model):
        means = await asyncio.gather(*[self._prompt_logprobs(model, record['prefix'] + t, len(t))
                                       for t in record['targets']])
        return torch.tensor([float(np.mean(m)) for m in means]).log_softmax(0).tolist()

    async def _generate(self, prefix, model, max_new, temperature, seed):
        sample = temperature > 0
        r = await self.client.completions.create(
            model=model, prompt=prefix, max_tokens=max_new, temperature=temperature, top_p=1.0,
            seed=int(seed) & 0x7FFFFFFF, logprobs=0 if sample else None,
            extra_body={'return_tokens_as_token_ids': True, 'skip_special_tokens': False})
        choice = r.choices[0]
        if choice.logprobs is not None and choice.logprobs.tokens:
            target = [int(t.split(':')[-1]) for t in choice.logprobs.tokens]
            old = [float(x) for x in choice.logprobs.token_logprobs]
        else:
            target = self.tok.encode(choice.text, add_special_tokens=False)
            old = None
        return target, old, choice.finish_reason == 'length'

    async def sync_weights(self):
        if self.version == 0:
            return
        name = f'actor-{self.session}-v{self.version}'
        path = os.path.join(self.lora_dir, name)
        os.makedirs(path, exist_ok=True)
        self.lm.save_pretrained(path)
        base = str(self.client.base_url).rstrip('/')
        r = await asyncio.to_thread(requests.post, f'{base}/load_lora_adapter',
                                    json=dict(lora_name=name, lora_path=os.path.abspath(path)), timeout=600)
        if r.status_code != 200:
            raise RuntimeError(f'load_lora_adapter failed ({r.status_code}): {r.text[:300]}')
        previous, self.adapter_name = self.adapter_name, name
        if previous:
            await asyncio.to_thread(requests.post, f'{base}/unload_lora_adapter', json=dict(lora_name=previous), timeout=120)
            shutil.rmtree(os.path.join(self.lora_dir, previous), ignore_errors=True)

    async def generate(self, messages, seed, tools=None, frozen=False, record_loss=True, max_new=None,
                       greedy=False, temperature=1.0):
        prefix = self.prompt(messages, tools)
        temperature = 0.0 if (frozen or greedy) else temperature
        max_new = max_new or self.args.max_new_tokens
        model = self.base_name if frozen else self._policy_model()
        target, old, truncated = await self._generate(prefix, model, max_new, temperature, seed)
        self.counts['frozen_tokens' if frozen else 'agent_tokens'] += len(target)
        text = self.tok.decode(target, skip_special_tokens=True).split('</think>')[-1].strip()
        record = None
        if record_loss and not frozen:
            if old is None or len(old) != len(target) or temperature != 1.0:
                old = await self._prompt_logprobs(self._policy_model(), prefix + target, len(target))
            ref = await self._prompt_logprobs(self.base_name, prefix + target, len(target))
            record = dict(kind='native', prefix=prefix, target=target, old=old, ref=ref,
                          version=self.version, truncated=truncated)
        return text, record

    async def propose(self, messages, asked):
        text = visible_text(messages)
        key = hashlib.sha256(text.encode()).hexdigest()
        m = int(self.args.topm)
        if key not in self.proposals:
            if self.args.dataset == 'ask_overconfidence':
                request = CLAIM_PROMPT.format(m=m) + text
            else:
                request = LIST_PROMPT.format(words=NUMBER_WORDS.get(m, str(m))) + text
            reply, _ = await self.generate([{'role': 'user', 'content': request}], 0, frozen=True, max_new=160)
            items = [re.sub(r'^\s*(\d+[.)]|[-*\u2022])\s*', '', s).strip() for s in reply.splitlines()]
            kept = [z for z in items if z and z.lower().strip('. ') not in EMPTY_ITEMS]
            self.proposals[key] = list(dict.fromkeys(kept))[:m]
        return [z for z in self.proposals[key] if z not in asked]

    def embed(self, text):
        key = hashlib.sha256(text.encode()).hexdigest()
        if key not in self.embeddings:
            ids = self.tok.encode(text, add_special_tokens=False)[-1536:] or [self.tok.eos_token_id]
            with self.lm.disable_adapter(), torch.no_grad():
                hidden = self.lm.get_base_model().model(input_ids=torch.tensor([ids], device=self.device),
                                                        use_cache=False).last_hidden_state[0, -1]
            self.embeddings[key] = hidden.float().cpu()
        return self.embeddings[key]

    def features(self, messages, labels):
        s = self.embed(visible_text(messages)).unsqueeze(0).to(self.device)
        z = torch.stack([self.embed('Information target: ' + t) for t in labels]).to(self.device)
        return s, z

    def values(self, messages, labels, budget):
        if not labels:
            return np.array([])
        s, z = self.features(messages, labels)
        with torch.no_grad():
            return self.head(s, z, budget).cpu().numpy().astype(float)

    def update_actor(self, weighted_records, n_groups):
        self.lm.train()
        self.opt.zero_grad(set_to_none=True)
        pg_sum = kl_sum = clipped = count = 0.0
        for rec, adv, weight in weighted_records:
            if rec['version'] != self.version:
                raise RuntimeError('stale rollout version')
            old = torch.tensor(rec['old'], device=self.device)
            ref = torch.tensor(rec['ref'], device=self.device)
            if rec['kind'] == 'menu':
                all_lp = self.menu_lp(rec)
                lp, old = all_lp[rec['action']], old[rec['action']]
                kl = (all_lp.exp() * (all_lp - ref)).sum()
            else:
                lp = self.token_lp(rec['prefix'], rec['target'])
                delta = ref - lp
                kl = (delta.exp() - delta - 1).sum()
            ratio = (lp - old).exp()
            pg = -torch.minimum(ratio * adv, ratio.clamp(1 - self.args.clip, 1 + self.args.clip) * adv).sum()
            n = max(1, ratio.numel())
            pg, kl = pg / n, kl / n
            loss = weight * (pg + self.args.kl_coef * kl) / n_groups
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite actor loss')
            loss.backward()
            pg_sum += float(pg.detach()) * weight / n_groups
            kl_sum += float(kl.detach()) * weight / n_groups
            clipped += float(((ratio - 1).abs() > self.args.clip).sum())
            count += ratio.numel()
        norm = float(torch.nn.utils.clip_grad_norm_(self.lm.parameters(), self.args.grad_clip))
        if not np.isfinite(norm):
            raise FloatingPointError('non-finite actor gradient')
        self.opt.step()
        self.lm.eval()
        self.version += 1
        return dict(pg_loss=pg_sum, reference_kl=kl_sum, grad_norm=norm,
                    clip_fraction=clipped / max(count, 1), loss_records=len(weighted_records))

    def update_head(self, groups):
        examples = []
        for g in groups:
            if not g['labels']:
                continue
            s, z = self.features(g['messages'], g['labels'])
            m = len(g['labels'])
            indices = [i for i in g['rewards'] if i < m]
            base = np.mean(g['rewards'][m])
            ys = torch.tensor([np.mean(g['rewards'][i]) - base for i in indices], device=self.device, dtype=torch.float32)
            examples.append((s, z, indices, ys, g['budget']))
        if not examples:
            return dict(controller_mse=None, controller_grad_norm=0.0)
        total = norm = 0.0
        for _ in range(self.args.controller_epochs):
            self.hopt.zero_grad(set_to_none=True)
            total = 0.0
            for s, z, idx, ys, b in examples:
                loss = (self.head(s, z, b)[idx] - ys).square().sum() / len(examples)
                loss.backward()
                total += float(loss.detach())
            norm = float(torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0))
            if not np.isfinite(total + norm):
                raise FloatingPointError('non-finite controller update')
            self.hopt.step()
        return dict(controller_mse=total, controller_grad_norm=norm)
