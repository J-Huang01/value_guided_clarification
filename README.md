# CLEAT

Code for CLEAT, a method that trains a language-model agent to ask clarifying questions together with a
lightweight controller that decides when a question is worth asking and what to ask about.

## Overview

The agent and the controller are trained jointly in a single online loop.

1. **Propose.** At each decision point the frozen backbone lists a few clarification targets.
2. **Contrast.** A rollout group compares asking about a target with letting the agent continue on its own.
   Exploration favours targets the controller currently rates highly.
3. **Fit the controller.** The controller regresses the residual value of asking, that is the return after
   asking minus the return without asking, under the current agent.
4. **Update the agent.** The agent is updated with advantages measured against the no-question continuations,
   so it learns from the same contrast the controller is fitted on.

At inference the controller asks about the highest-value target whenever that value is positive, and otherwise
lets the agent continue on its own for the rest of the episode.

## Requirements

```bash
pip install -r requirements.txt
```

The method uses two language models: the agent that is trained and a user simulator that also acts as judge.
Both are served through an OpenAI-compatible endpoint.

## Benchmarks

Get each benchmark from its page below and place it under `third_party/` (for example with `git clone <link> third_party/<name>`).

| Benchmark | `--dataset` | Download |
|---|---|---|
| AskMind | `ask_mind` | [link](https://github.com/jialeuuz/askbench) |
| AskOverconfidence | `ask_overconfidence` | [link](https://github.com/jialeuuz/askbench) |
| UserGym | `usergym` | [link](https://github.com/SalesforceAIResearch/UserRL) |
| IN3 | | [link](https://huggingface.co/datasets/hbx/IN3) |
| tau2-bench | | [link](https://github.com/sierra-research/tau2-bench) |

For datasets distributed as a task pool and a test file, `cleat/make_splits.py` builds the train, development
and test splits and removes pool tasks that also appear in the test set.

## Quick start

`run.sh` serves both models with vLLM, trains, and evaluates the final checkpoint on the test set with and
without the controller:

```bash
DATASET=<dataset> AGENT_MODEL=<agent model> USER_MODEL=<simulator model> bash run.sh
```

Extra arguments are passed to `cleat/train.py`; devices, ports and extra server flags can be set through the
variables at the top of `run.sh`. The defaults are the settings reported in the paper;
`python cleat/train.py --help` lists every option. To use endpoints that are already running, call
`cleat/train.py` and `cleat/evaluate.py` directly with `--agent-url` and `--user-url`.

## Outputs

| Path | Content |
|---|---|
| `runs/<dataset>/train.jsonl` | one row per training step: returns, residual query value, losses and interaction counts |
| `runs/<dataset>/eval.jsonl` | development-set episodes during training, with and without the controller |
| `runs/<dataset>/checkpoints/` | `warmup`, periodic `step-XXXX` and the deployed `final-refit` checkpoint |
| `runs/<dataset>/test/summary.json` | test metrics for each seed and their mean |

## Code structure

| File | Role |
|---|---|
| `run.sh` | serve, train and evaluate in one command |
| `cleat/train.py` | online co-training loop |
| `cleat/evaluate.py` | test-set evaluation |
| `cleat/actor.py` | agent policy, controller head and their updates |
| `cleat/math_core.py` | guided sampling, inclusion probabilities, advantages and the decision rule |
| `cleat/environment.py` | benchmark interaction and task splits |
| `cleat/benchmarks/` | benchmark-specific task loading, judge and user simulation |
