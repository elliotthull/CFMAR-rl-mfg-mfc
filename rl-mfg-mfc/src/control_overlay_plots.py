import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Theoretical controls (shared by both notebooks) ──────────────────────────
def op_control_0(x):
    return -(1.309571555*x + (2.309085513 - 1.309571555)*0.5)

def op_control_7(x):
    return -(1.083475891*x + (1.737935738 - 1.083475891)*(0.5*np.power(np.e, -0.760346885375)))

def op_control_15(x):
    return -(0.4168792984*x + (0.4572735727 - 0.4168792984)*(0.5*np.power(np.e, -0.42869397440625)))

theoretical_controls = [op_control_0, op_control_7, op_control_15]
target_times = [0, 7, 15]
labels = ["t=0", "t=7/16", "t=15/16"]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — Idealized MFG trajectory relaxation  (idealizedattempt2.ipynb)
# ─────────────────────────────────────────────────────────────────────────────
print("Running Model 1: Idealized MFG...")

time_step         = 1 / 16
state_action_step = 1 / 8
time = np.arange(0, 1, time_step)
T    = len(time)

state_space_mfg  = np.arange(-2,  2.00, state_action_step, dtype=float)
action_space_mfg = np.arange(-3,  1.25, state_action_step, dtype=float)
num_X = len(state_space_mfg)
num_A = len(action_space_mfg)

c_alpha = 1.0; x_state = 2.0; gamma = 1.75; c_g = 0.3; x_vol = 0.5; dt = time_step
phi = 500; n_iter = 5000; n_Q_iter = 20; om_q = 0.55; om_mu = 0.85

std = x_vol * np.sqrt(dt)
P   = np.zeros((num_X, num_A, num_X))
for xi, x in enumerate(state_space_mfg):
    for ai, a in enumerate(action_space_mfg):
        mean    = x + a * dt
        weights = np.exp(-0.5 * ((state_space_mfg - mean) / std) ** 2)
        P[xi, ai, :] = weights / weights.sum()

def running_cost_matrix(theta_t):
    Ea = np.dot(action_space_mfg, theta_t)
    ca = 0.5 * c_alpha * action_space_mfg ** 2
    cx = 0.5 * x_state * state_space_mfg  ** 2
    F  = ca[None, :] + cx[:, None] - gamma * state_space_mfg[:, None] * Ea
    return F * dt

def terminal_cost_vector():
    return 0.5 * c_g * state_space_mfg ** 2

def soft_min_matrix(Q_t, phi):
    z_s = Q_t - Q_t.min(axis=1, keepdims=True)
    w   = np.exp(-phi * z_s)
    return w / w.sum(axis=1, keepdims=True)

def backward_Q_step(mu, Q, current_rho):
    Q_new = Q.copy()
    g     = terminal_cost_vector()
    for t in range(T - 1, -1, -1):
        pi = soft_min_matrix(Q[t], phi)
        theta_t = mu[t] @ pi
        F = running_cost_matrix(theta_t)
        V_next = Q[t+1].min(axis=1) if t < T-1 else g
        EV     = np.einsum('xaz,z->xa', P, V_next)
        Q_new[t] = Q[t] + current_rho * (F + EV - Q[t])
    return Q_new

def mu_operator(mu, Q):
    mu_new = np.zeros_like(mu)
    for t in range(T-1):
        pi = soft_min_matrix(Q[t], phi)
        muP = np.einsum('x,xa,xaz->z', mu[t], pi, P)
        mu_new[t+1] = muP - mu[t+1]
    return mu_new

mu_0_raw = np.exp(-0.5 * ((state_space_mfg - 0.5) / 0.3) ** 2)
mu_0  = mu_0_raw / mu_0_raw.sum()
mu_mfg = np.tile(mu_0, (T, 1))
Q_mfg  = np.zeros((T, num_X, num_A))

for n in range(1, n_iter + 1):
    rho_Q  = 1 / (1 + n)**om_q
    rho_mu = 1 / (1 + n)**om_mu
    for _ in range(n_Q_iter):
        Q_mfg = backward_Q_step(mu_mfg, Q_mfg, rho_Q)
    mu_mfg = mu_mfg + rho_mu * mu_operator(mu_mfg, Q_mfg)
    mu_mfg = np.clip(mu_mfg, 1e-12, None)
    mu_mfg = mu_mfg / mu_mfg.sum(axis=1, keepdims=True)
    if n % 400 == 0:
        print(f"  MFG iter {n}/{n_iter}")

# Extract MFG controls
mfg_controls = {}
for t_idx in target_times:
    mfg_controls[t_idx] = np.array([
        action_space_mfg[np.argmin(Q_mfg[t_idx, xi, :])]
        for xi in range(num_X)
    ])

print("Model 1 done.")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — Online RL Q-learning  (priceimpactmodel.ipynb)
# ─────────────────────────────────────────────────────────────────────────────
print("Running Model 2: RL Q-learning (1M episodes)...")

state_space_rl  = np.arange(-1.5, 2.00, state_action_step, dtype=float)
action_space_rl = np.arange(-2.5,  1.25, state_action_step, dtype=float)

Q_rl = np.zeros([T, len(state_space_rl), len(action_space_rl)], dtype=float)
num_episodes = 1_000_000
eps_greedy   = 0.1
x_vol_rl = 0.5; sigma_0 = 0.5; om_q_rl = 0.55; om_mu_rl = 0.85; c_g_rl = 0.3

mu_rl = np.ones([T, len(state_space_rl), len(action_space_rl)]) / (len(state_space_rl) * len(action_space_rl))
count_txa = np.zeros([T, len(state_space_rl), len(action_space_rl)])

def state_discretize_rl(x):
    return int(np.clip(np.argmin(np.abs(state_space_rl - x)), 0, len(state_space_rl) - 1))

def state_transition_rl(x, alpha):
    dt = 1/16
    return x + alpha*dt + x_vol_rl*np.sqrt(dt)*np.random.randn()

def get_mean_action_rl(mu, t_idx):
    marginal = np.sum(mu[t_idx], axis=0)
    return np.dot(action_space_rl, marginal)

def immediate_costs_rl(x_val, alpha_val, t_idx, mu):
    dt = 1/16
    mean_alpha = get_mean_action_rl(mu, t_idx)
    cost = (0.5*c_alpha*alpha_val**2 + 0.5*x_state*x_val**2) - gamma*x_val*mean_alpha
    return cost * dt

def epsilon_greedy_rl(q_values):
    if np.random.rand() < eps_greedy:
        return np.random.randint(0, len(action_space_rl))
    return int(np.argmin(q_values))

jump = round(num_episodes / 200)
for k in range(1, num_episodes + 1):
    x = np.random.uniform(state_space_rl.min(), state_space_rl.max())
    for t_idx in range(T):
        x_idx      = state_discretize_rl(x)
        action_idx = epsilon_greedy_rl(Q_rl[t_idx, x_idx, :])
        alpha_val  = action_space_rl[action_idx]
        count_txa[t_idx, x_idx, action_idx] += 1
        rho_Q = 1 / (1 + count_txa[t_idx, x_idx, action_idx])**om_q_rl
        rho_Mu = 1 / (1 + k)**om_mu_rl
        delta = np.zeros((len(state_space_rl), len(action_space_rl)))
        delta[x_idx, action_idx] = 1.0
        mu_rl[t_idx] += rho_Mu * (delta - mu_rl[t_idx])

        cost_t = immediate_costs_rl(x, alpha_val, t_idx, mu_rl)
        next_x = np.clip(state_transition_rl(x, alpha_val), state_space_rl.min(), state_space_rl.max())
        x_next_idx = state_discretize_rl(next_x)

        if t_idx < T - 1:
            td_target = cost_t + np.min(Q_rl[t_idx + 1, x_next_idx, :])
        else:
            td_target = cost_t + (c_g_rl / 2) * next_x**2

        Q_rl[t_idx, x_idx, action_idx] += rho_Q * (td_target - Q_rl[t_idx, x_idx, action_idx])
        x = next_x

    if k % 200000 == 0:
        print(f"  RL episode {k}/{num_episodes}")

# Extract RL controls
rl_controls = {}
for t_idx in target_times:
    rl_controls[t_idx] = np.array([
        action_space_rl[np.argmin(Q_rl[t_idx, xi, :])]
        for xi in range(len(state_space_rl))
    ])

print("Model 2 done.")

# ─────────────────────────────────────────────────────────────────────────────
# OVERLAY PLOT
# ─────────────────────────────────────────────────────────────────────────────
x_range = np.linspace(-2, 2, 1000)
colors   = ['#1f77b4', '#ff7f0e', '#2ca02c']   # one per time slice

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)


for i, t_idx in enumerate(target_times):
    ax = axes[i]

    # Theoretical
    ax.plot(x_range, theoretical_controls[i](x_range),
            color='red', linewidth=2.5, label='Theoretical', zorder=5)

    # MFG
    ax.scatter(state_space_mfg, mfg_controls[t_idx],
               marker='x', s=40, color='royalblue', alpha=0.85,
               label='Idealized MFG', zorder=4)

    # RL
    ax.scatter(state_space_rl, rl_controls[t_idx],
               marker='*', s=40, color='darkorange', alpha=0.85,
               label='RL Q-Learning', zorder=3)

    ax.set_title(f"Controls at {labels[i]}", fontsize=12)
    ax.set_xlabel("State x", fontsize=11)
    if i == 0:
        ax.set_ylabel(r"$\alpha(t, x)$", fontsize=12)
    ax.set_xlim(-1.75, 1.8)
    ax.set_ylim(-3, 5.25)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

plt.tight_layout()
out_path = "/Users/elliotthull/Desktop/control_overlay.png"
plt.savefig(out_path, dpi=600, bbox_inches='tight')
print(f"Saved to {out_path}")
plt.show()
