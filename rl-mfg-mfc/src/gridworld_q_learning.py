import numpy as np
import random
import matplotlib as plt

X = np.arange(16, dtype=int)
A = np.array(['Up', 'Down', 'Right', 'Left'])
Q = np.zeros((len(X), len(A)))

X_n_1 = np.array([
    # action indexes 0:Up, 1:Right, 2:Down, 3:Left
    # State 0 
    [0, 1, 4, 0], 
    # State 1 
    [1, 2, 5, 0],
    # State 2 
    [2, 3, 6, 1], 
    # State 3
    [3, 3, 7, 2],
    # State 4 
    [0, 5, 8, 4],
    # State 5 
    [1, 6, 9, 4], 
    # State 6
    [2, 7, 10, 5],
    # State 7 
    [3, 7, 11, 6], 
    # State 8
    [4, 9, 12, 8],
    #State 9
    [5,10,13,8],
    #State 10
    [6,11,14,9],
    #State 11
    [7,11,15,10],
    #State 12
    [8,13,12,12],
    #State 13
    [9,14,13,12],
    #State 14
    [14,14,14,14],
    #State 15
    [11,15,15,14]
])

GOAL_STATE = 14
REWARD_FOR_GOAL = 10.0

def get_reward(current_state, action):
    """Calculates the immediate reward after taking an action."""
    def cost_function(current_state):

        if current_state in [1,3,4,9,12]:
            return  -1
        if current_state in [2,5,8,10,15]:
            return -2
        else:
            return -4


    next_step = X_n_1[current_state, action]
    if next_step == GOAL_STATE:
        return REWARD_FOR_GOAL
    else:
        return cost_function(current_state)

#parameters
LEARNING_RATE = 0.9
DISCOUNT_FACTOR = 0.9
EPISODES = 5000      
MAX_STEPS = 20          

#Epsilon-Greedy Parameters
epsilon = 1.0           
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.999   

#training loop
for episode in range(EPISODES):
    # Start every episode at State 0 (Top-Left)
    current_state = 0
    done = False
    
    for step in range(MAX_STEPS):
        
        #action selection
        if random.random() < epsilon:
            action = random.randint(0, len(A) - 1)
        else:
            action = np.argmax(Q[current_state, :])
        
        #apply action
        reward = get_reward(current_state, action)
        next_state = X_n_1[current_state, action]
        
        if next_state == GOAL_STATE:
            done = True
        
        #update Q table
        if done:
            max_future_q = 0.0
        else:
            max_future_q = np.max(Q[next_state, :])

        old_q = Q[current_state, action]
        Q[current_state, action] = old_q + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * max_future_q - old_q
        )
        
        #update state
        current_state = next_state
        
        if done:
            break
            
    #decay 
    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

#Results
print("Training Complete!")

print(Q)
print("-" * 30)

#Optimal policy
policy = np.argmax(Q, axis=1)
action_map = {0: "Up", 1: "Right", 2: "Down", 3: "Left", 4: "Reached Destination"}

print("Optimal Policy:")
for s in range(len(X)):
    print(f"State {s}: {action_map[policy[s]]}")

