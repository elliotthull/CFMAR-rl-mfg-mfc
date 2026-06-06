import numpy as np
import random
import matplotlib.pyplot as plt

# Parameters
states = np.array([0, 1])
actions = np.array([0, 1])  # 0 = stay, 1 = move
N = 100000  # number of steps
phi = 500  # soft-min parameter
p = 0.01
c = 5
gamma = 0.5
tol_q = 0.1
tol_mu = 0.01
om_q = 0.85
om_mu = 0.55

# Initialize Q and mu as lists (storing history)
Q = [np.zeros((len(states), len(actions)))]
# mu[s,a] = occupancy measure (probability of being in state s and taking action a)
mu = [np.ones((len(states), len(actions))) / 4]  # Initial uniform distribution

# Track visit counts for learning rate
count = np.zeros([2, 2])

def rhosCalc(count_xa, n):
    """Calculate learning rates for Q and mu updates"""
    rhoQ = 1 / np.power(1 + count_xa, om_q)
    rhoMu = 1 / np.power(2 + n, om_mu)
    return {'q': rhoQ, 'mu': rhoMu}

def env(state, action, mu_dist):
    """Environment transition function"""
    if random.random() < p:
        newS = (state + action) % 2
    else: 
        newS = (state + action - 1) % 2
    
    # Cost depends on state and congestion at next state-action
    # Sum over all actions at next state weighted by mu
    congestion = np.sum(mu_dist[newS, :])
    cost = c * state + congestion
    return {"newState": newS, "cost": cost}

def softminAct(Q_x):
    """Select action using soft-min policy"""
    z = Q_x - np.min(Q_x)            # numerical-stability shift: identical probabilities, avoids exp() overflow
    weights = np.exp(-phi * z)
    weights = weights / np.sum(weights)
    return np.random.choice(actions, p=weights)

def getSoftminPolicy(Q_x):
    """Get soft-min policy probabilities"""
    z = Q_x - np.min(Q_x)            # numerical-stability shift: identical probabilities, avoids exp() overflow
    weights = np.exp(-phi * z)
    return weights / np.sum(weights)

def computeStateDist(mu_xa):
    """Compute marginal state distribution from (state,action) distribution"""
    return np.sum(mu_xa, axis=1)

def computeNextMu(mu_current, Q_current):
    """Compute next occupancy measure under current policy (CONTROL)"""
    mu_next = np.zeros((len(states), len(actions)))
    
    # Get marginal state distribution
    mu_state = computeStateDist(mu_current)
    
    for x in states:
        # Get policy at state x
        pi = getSoftminPolicy(Q_current[x])
        
        # For each action, compute flow into next states
        for a in actions:
            # Transitions from state x with action a
            next_state_p = (x + a) % 2
            next_state_1mp = (x + a - 1) % 2
            
            # Flow into next state-action pairs
            for a_next in actions:
                pi_next_p = getSoftminPolicy(Q_current[next_state_p])
                pi_next_1mp = getSoftminPolicy(Q_current[next_state_1mp])
                
                # Probability flow: current state -> action -> next state -> next action
                mu_next[next_state_p, a_next] += mu_state[x] * pi[a] * p * pi_next_p[a_next]
                mu_next[next_state_1mp, a_next] += mu_state[x] * pi[a] * (1 - p) * pi_next_1mp[a_next]
    
    return mu_next

# Randomly choose initial state based on initial mu
mu_state_init = computeStateDist(mu[0])
x = np.random.choice(states, p=mu_state_init)

# Iterate N times
for n in range(N):
    # Copy new Q matrix, mu distribution
    Q.append(np.copy(Q[n]))
    mu.append(np.copy(mu[n]))
    
    # Use softmin to choose next action
    a = softminAct(Q[n + 1][x])
    
    # Update counter for rho calculation
    count[x][a] += 1
    rhos = rhosCalc(count[x][a], n)
    
    # Input state, action, mu into environment
    # and receive new state and cost
    envir = env(x, a, mu[n + 1])
    
    # Update Q matrix
    Q[n + 1][x][a] += rhos['q'] * (envir["cost"] + gamma * np.min(Q[n + 1][envir["newState"]]) - Q[n + 1][x][a])
    
    # Move to next state
    x = envir["newState"]
    
    # MEAN FIELD CONTROL UPDATE: 
    # Update mu based on expected flow under current policy
    mu_expected = computeNextMu(mu[n], Q[n + 1])
    mu[n + 1] = mu[n] + rhos['mu'] * (mu_expected - mu[n])
    
    # Normalize to maintain probability distribution
    mu[n + 1] = mu[n + 1] / np.sum(mu[n + 1])

# Visualization of convergence
abridgedmu = []
for i in range(0, N, 100):
    abridgedmu.append(mu[i][0, 0])  # Track mu[state=0, action=0]

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(list(range(0, N, 100)), abridgedmu)
plt.xlabel('Iteration')
plt.ylabel('mu[0,0] (State 0, Stay)')
plt.title('Occupancy Measure mu[0,0] Convergence')
plt.grid(True)

# Plot all mu components
plt.subplot(2, 2, 2)
for s in states:
    for a in actions:
        mu_trace = [mu[i][s, a] for i in range(0, N, 100)]
        plt.plot(list(range(0, N, 100)), mu_trace, label=f'mu[{s},{a}]')
plt.xlabel('Iteration')
plt.ylabel('Occupancy Measure')
plt.title('All Occupancy Measures')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
state_dist = [computeStateDist(mu[i])[0] for i in range(0, N, 100)]
plt.plot(list(range(0, N, 100)), state_dist)
plt.xlabel('Iteration')
plt.ylabel('Marginal Probability')
plt.title('Marginal State Distribution (State 0)')
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nFinal Q-table:")
print(Q[-1])
print("\nFinal occupancy measure mu:")
print(mu[-1])
print("\nMarginal state distribution:")
print(computeStateDist(mu[-1]))
print("\nFinal policies:")
for s in states:
    pi = getSoftminPolicy(Q[-1][s])
    print(f"State {s}: stay={pi[0]:.4f}, move={pi[1]:.4f}")
    
