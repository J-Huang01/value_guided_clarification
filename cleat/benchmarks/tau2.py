import asyncio
import json
import re

from benchmarks import JudgeFailure
from benchmarks.askbench import canonical_question

TAU2_DOMAINS = ('airline', 'retail', 'telecom')
MAX_STEPS = 100
TOOL_CALL = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.S)


def load_tasks(domain, split):
    from tau2.runner.helpers import get_tasks
    return [dict(env_name=domain, gold=str(t.id), task=t) for t in get_tasks(domain, task_split_name=split)]


def _clean(messages):
    out = []
    for m in messages:
        m = dict(m)
        m['content'] = m.get('content') or ''
        if not m.get('tool_calls'):
            m.pop('tool_calls', None)
        out.append(m)
    return out


def _parse_calls(text):
    from tau2.data_model.message import ToolCall
    calls = []
    for i, block in enumerate(TOOL_CALL.findall(text)):
        try:
            call = json.loads(block)
            arguments = call.get('arguments') or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append(ToolCall(id=f'call_{i}', name=str(call['name']), arguments=dict(arguments)))
        except (ValueError, KeyError, TypeError):
            continue
    return calls


class EpisodeRunner:
    def __init__(self, tau2, loop, seed, first_label=None, controller=None, max_queries=0, record_loss=False,
                 greedy=False, temperature=1.0):
        self.tau2, self.loop, self.seed = tau2, loop, seed
        self.first_label, self.controller, self.max_queries = first_label, controller, max_queries
        self.record_loss, self.greedy, self.temperature = record_loss, greedy, temperature
        self.first = True
        self.stopped = False
        self.asked = []
        self.records = []
        self.queries = self.native_queries = self.turns = 0
        self.trace = []
        self.schemas = None

    def respond(self, messages, from_user):
        return asyncio.run_coroutine_threadsafe(self._respond(messages, from_user), self.loop).result()

    async def _respond(self, messages, from_user):
        from tau2.data_model.message import AssistantMessage
        actor = self.tau2.actor
        messages = _clean(messages)
        dialogue = [m for m in messages if m['role'] != 'system']
        label = None
        if self.first and self.first_label is not None:
            label = self.first_label
        elif from_user and self.controller is not None and not self.stopped and self.queries < self.max_queries:
            label = await self.controller(dialogue, self)
        self.first = False
        self.turns += 1
        actor.counts['environment_steps'] += 1
        if label is not None:
            self.queries += 1
            self.asked.append(label)
            text = canonical_question(label)
            self.trace.append(dict(turn=self.turns, semantic_query=label, text=text))
            return AssistantMessage(role='assistant', content=text)
        seed = (self.seed * 1000003 + self.turns) & 0x7FFFFFFF
        text, record = await actor.generate(messages, seed, tools=self.schemas, record_loss=self.record_loss,
                                            greedy=self.greedy, temperature=self.temperature)
        if record:
            self.records.append(record)
        calls = _parse_calls(text)
        content = text.split('<tool_call>')[0].strip()
        if not calls and '?' in content:
            self.native_queries += 1
        self.trace.append(dict(turn=self.turns, semantic_query=None, text=text))
        if calls:
            return AssistantMessage(role='assistant', content=content or None, tool_calls=calls)
        return AssistantMessage(role='assistant', content=content or '...')


def _agent_class():
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import MultiToolMessage, UserMessage
    from tau2.utils.llm_utils import to_litellm_messages

    class CleatAgent(LLMAgent):
        def __init__(self, tools, domain_policy, runner):
            super().__init__(tools=tools, domain_policy=domain_policy, llm='cleat', llm_args={})
            self.runner = runner
            runner.schemas = [tool.openai_schema for tool in tools]

        def _generate_next_message(self, message, state):
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)
            history = to_litellm_messages(state.system_messages + state.messages)
            return self.runner.respond(history, isinstance(message, UserMessage))
    return CleatAgent


class Tau2:
    def __init__(self, actor, user_url, user_model, user_temperature=0.7):
        self.actor = actor
        self.user_llm = 'openai/' + user_model
        self.user_args = dict(api_base=user_url, api_key='EMPTY', temperature=user_temperature)

    def _user(self, environment, task, seed):
        from tau2.runner.build import build_user
        user = build_user('user_simulator', environment, task, llm=self.user_llm, llm_args=dict(self.user_args))
        user.set_seed(seed)
        return user

    def _root_sync(self, task, seed):
        try:
            return self._opening(task, seed)
        except Exception as error:
            raise JudgeFailure(f'tau2 opening failed: {error!r}') from error

    def _opening(self, task, seed):
        from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
        from tau2.runner.build import build_environment
        environment = build_environment(task['env_name'])
        user = self._user(environment, task['task'], seed)
        greeting = DEFAULT_FIRST_AGENT_MESSAGE.model_copy(deep=True)
        reply, _ = user.generate_next_message(greeting, user.get_init_state())
        self.actor.counts['simulator_calls'] += 1
        return [greeting, reply]

    async def root(self, task, seed):
        from tau2.utils.llm_utils import to_litellm_messages
        history = await asyncio.to_thread(self._root_sync, task, seed)
        return dict(task=task, history=history, messages=_clean(to_litellm_messages(history)))

    def _run_sync(self, task, history, runner, seed):
        try:
            return self._simulate(task, history, runner, seed)
        except Exception as error:
            raise JudgeFailure(f'tau2 episode failed: {error!r}') from error

    def _simulate(self, task, history, runner, seed):
        from tau2.data_model.tasks import InitialState
        from tau2.orchestrator.orchestrator import Orchestrator
        from tau2.runner.build import build_environment
        from tau2.runner.simulation import run_simulation
        official = task['task']
        if history is not None:
            state = official.initial_state
            official = official.model_copy(update=dict(initial_state=InitialState(
                initialization_data=state.initialization_data if state else None,
                initialization_actions=state.initialization_actions if state else None,
                message_history=[m.model_copy(deep=True) for m in history])))
        environment = build_environment(task['env_name'])
        agent = _agent_class()(environment.get_tools(), environment.get_policy(), runner)
        user = self._user(environment, official, seed)
        orchestrator = Orchestrator(domain=task['env_name'], agent=agent, user=user, environment=environment,
                                    task=official, max_steps=MAX_STEPS, seed=seed)
        simulation = run_simulation(orchestrator)
        self.actor.counts['simulator_calls'] += sum(1 for m in simulation.messages if getattr(m, 'role', '') == 'user')
        return float(simulation.reward_info.reward) if simulation.reward_info else 0.0

    def _result(self, task, runner, reward):
        return dict(uid=task['uid'], env_name=task['env_name'], pass1=reward, reward=reward, correct=None,
                    coverage=None, info=None, queries=runner.queries, native_queries=runner.native_queries,
                    turns=runner.turns, trace=runner.trace)

    async def rollout(self, root, label, seed, record_loss=True):
        runner = EpisodeRunner(self, asyncio.get_running_loop(), seed, first_label=label, record_loss=record_loss)
        reward = await asyncio.to_thread(self._run_sync, root['task'], root['history'], runner, seed)
        return dict(result=self._result(root['task'], runner, reward), records=runner.records, after_first=None)

    async def episode(self, task, seed, controller=None, max_queries=0, greedy=False, temperature=1.0):
        runner = EpisodeRunner(self, asyncio.get_running_loop(), seed, controller=controller, max_queries=max_queries,
                               greedy=greedy, temperature=temperature)
        reward = await asyncio.to_thread(self._run_sync, task, None, runner, seed)
        return self._result(task, runner, reward)
