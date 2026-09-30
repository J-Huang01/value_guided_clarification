import asyncio
import contextvars
import json
import os
import re

import pandas as pd
import yaml
from openai import AsyncOpenAI

TRAVEL_ENVS = ['travel22', 'travel33', 'travel44', 'travel233', 'travel333', 'travel334', 'travel444', 'travel2222']
TRAIN_ENVS = TRAVEL_ENVS + ['turtle']
HELD_OUT_ENVS = ['intention', 'telepathy']
TITLE_KEYED = ('turtle', 'telepathy')
EPISODE_STEPS = 16
_errors = contextvars.ContextVar('simulator_errors', default=None)


def _data_path(userrl_root, env_name, split):
    if 'travel' in env_name:
        return os.path.join(userrl_root, 'data', f'{env_name}_multiturn_onechoice', f'{split}.parquet')
    return os.path.join(userrl_root, 'data', f'{env_name}_multiturn', f'{split}.parquet')


def load_tasks(userrl_root, env_name, split):
    df = pd.read_parquet(_data_path(userrl_root, env_name, split))
    tasks = []
    for i in range(len(df)):
        reward_model = dict(df.iloc[i]['reward_model'])
        gold = reward_model.get('title') if env_name in TITLE_KEYED else reward_model.get('id', i)
        tasks.append(dict(env_name=env_name, gold=str(gold), messages=[dict(m) for m in list(df.iloc[i]['prompt'])]))
    return tasks


def tool_schema(userrl_root):
    with open(os.path.join(userrl_root, 'eval', 'schema', 'interact_tool.yaml')) as f:
        return yaml.safe_load(f)['tool_schema']


def flatten(messages):
    parts = []
    for m in messages:
        content = m.get('content')
        if isinstance(content, list):
            content = json.dumps(content)
        if m.get('role') == 'assistant' and m.get('tool_calls'):
            try:
                a = json.loads(m['tool_calls'][0]['function']['arguments'])
                content = f"[{a.get('choice')}] {a.get('content')}"
            except (ValueError, KeyError, TypeError, IndexError):
                pass
        parts.append(f"{m.get('role')}: {content}")
    return '\n'.join(parts)


def is_question(choice, content):
    if choice != 'action':
        return False
    text = (content or '').lower()
    return '?' in text or bool(re.search(r'\b(what|which|could you|do you|would you|any|prefer|how|when|where)\b', text))


def parse_tool_call(text):
    match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.S)
    try:
        call = json.loads(match.group(1)) if match else None
        if call and call.get('name') == 'interact_with_env':
            args = call['arguments']
            if isinstance(args, str):
                args = json.loads(args)
            return args['choice'], str(args['content']), call
    except (ValueError, KeyError, TypeError):
        pass
    return 'action', text, None


class UserGym:
    def __init__(self, userrl_root, user_url, user_model, counts):
        self.root = userrl_root
        self.user_url = user_url
        self.user_model = user_model
        self.counts = counts
        self.schema = tool_schema(userrl_root)
        os.environ.setdefault('OPENAI_API_KEY', 'EMPTY')
        os.environ.setdefault('OPENAI_BASE_URL', user_url)
        self._patch_travel()
        self._patch_turtle()

    def _patch_travel(self):
        import travelgym.env.prompt_async as travel_prompts
        client = AsyncOpenAI(api_key='EMPTY', base_url=self.user_url, max_retries=0)
        counts = self.counts

        async def call(system_prompt, user_prompt, model_config):
            last_error = None
            for attempt in range(3):
                try:
                    counts['simulator_calls'] += 1
                    response = await client.chat.completions.create(
                        model=model_config['model_name'], temperature=model_config['temperature'],
                        max_tokens=model_config['max_tokens'], timeout=model_config['timeout'],
                        messages=[dict(role='system', content=system_prompt), dict(role='user', content=user_prompt)])
                    result = travel_prompts.parse_output_as_json(response.choices[0].message.content or '')
                    if not isinstance(result, dict):
                        raise ValueError('travel simulator did not return a JSON object')
                    return result
                except Exception as error:
                    last_error = repr(error)
                    if attempt < 2:
                        await asyncio.sleep(2)
            errors = _errors.get()
            if errors is not None:
                errors.append('travel simulator exhausted retries: ' + str(last_error))
            return None
        travel_prompts.async_model_call = call

    def _patch_turtle(self):
        import turtlegym.env.story_env as story_env
        if getattr(story_env, '_cleat_patched', False):
            return
        original = story_env.evaluate_action_async
        counts = self.counts

        async def wrapped(*a, **kw):
            last_error = None
            for attempt in range(3):
                try:
                    counts['simulator_calls'] += 1
                    return await original(*a, **kw)
                except Exception as error:
                    last_error = repr(error)
                    if attempt < 2:
                        await asyncio.sleep(2)
            raise RuntimeError('turtle simulator exhausted retries: ' + str(last_error))
        story_env.evaluate_action_async = wrapped
        story_env._cleat_patched = True

    def build_env(self, task, max_steps):
        name = task['env_name']
        if 'travel' in name:
            import travelgym
            config = travelgym.get_default_config()
            config.one_choice_per_aspect = True
            config.search_correct_reward = 0.2
            config.preference_correct_reward = 0.6
            env_class = travelgym.TravelEnv
        elif name == 'turtle':
            import turtlegym
            config = turtlegym.get_default_config()
            config.success_threshold = 1.0
            env_class = turtlegym.StoryEnv
        elif name == 'intention':
            import intentiongym
            config = intentiongym.get_default_config()
            env_class = intentiongym.IntentionEnv
        elif name == 'telepathy':
            import telepathygym
            config = telepathygym.get_default_config()
            env_class = telepathygym.TelepathyEnv
        else:
            raise ValueError('unsupported UserGym environment: ' + name)
        config.max_steps = max_steps
        config.base_url = self.user_url
        config.timeout = 120
        config.data_mode = 'single'
        config.data_source = task['gold']
        config.model_name = self.user_model
        env = env_class(config=config)
        env.reset()
        return env

    async def env_step(self, env, action):
        errors = []
        token = _errors.set(errors)
        try:
            result = await env.step_async(action)
            if errors:
                raise RuntimeError('; '.join(errors))
            return result
        finally:
            _errors.reset(token)
