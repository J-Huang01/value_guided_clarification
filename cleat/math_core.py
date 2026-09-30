import numpy as np


def probabilities(pi, values, beta, epsilon):
    pi = np.asarray(pi, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    m = len(values)
    if m == 0:
        return np.array([]), np.array([]), np.array([1.0])
    q = pi[:-1] / pi[:-1].sum()
    bar = (1 - epsilon) * q + epsilon / m
    logits = np.log(bar) + beta * values
    mu = np.exp(logits - logits.max())
    mu /= mu.sum()
    rho = mu.copy()
    if m > 1:
        for j in range(m):
            for z in range(m):
                if z != j:
                    rho[z] += mu[j] * bar[z] / (1 - bar[j])
    else:
        rho[:] = 1
    return bar, mu, np.r_[rho, 1.0]


def guidance(values, scale, clip=6.0):
    v = np.asarray(values, dtype=np.float64)
    if v.size:
        v = v - v.mean()
    return np.clip(v * float(scale), -clip, clip)


def sample_arms(pi, values, beta, epsilon, rng, n_pass=2, n_guided=2, n_cover=1):
    if n_pass < 2:
        raise ValueError('the PASS reference needs at least two continuations')
    bar, mu, rho = probabilities(pi, values, beta, epsilon)
    m = len(values)
    arms = {m: n_pass}
    if m:
        guided = int(rng.choice(m, p=mu))
        arms[guided] = n_guided
        if m > 1:
            cover = bar.copy()
            cover[guided] = 0
            cover /= cover.sum()
            arms[int(rng.choice(m, p=cover))] = n_cover
        else:
            arms[guided] += n_cover
    return arms, rho, bar, mu


def advantages(rewards, pass_index):
    pass_rewards = np.asarray(rewards[pass_index], dtype=float)
    if len(pass_rewards) < 2:
        raise ValueError('the PASS reference needs at least two continuations')
    out = {}
    for arm, rs in rewards.items():
        rs = np.asarray(rs, dtype=float)
        if arm == pass_index:
            out[arm] = rs - (pass_rewards.sum() - pass_rewards) / (len(pass_rewards) - 1)
        else:
            out[arm] = rs - pass_rewards.mean()
    return out


def effective_sample_size(weights):
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / max(float((w * w).sum()), 1e-30))


def value_rule(values):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return 0
    i = int(np.argmax(v))
    return i if v[i] > 0.0 else len(v)
