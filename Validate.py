import numpy as np
import pandas as pd

import torch

from matplotlib import pyplot as plt

from collections import deque

from pid import PIDModel
from agent import Agent, Actor, Critic, Transition

T_SIZE = 100
SET_POINT = 1000

t = np.linspace(0, 500, num=T_SIZE)
SP = np.ones(T_SIZE)*SET_POINT
#SP[0: 5] = 0
#SP[5: 25] = 500
#SP[26: 50] = 1500
#SP[51:75 ] = 1750
#SP[76:99 ] = 2000

env = PIDModel(ku=1.396, tu=3.28, t=t, SP=SP)

actor = Actor()
critic = Critic()
agent = Agent(env,
    actor_lr=0, critic_lr=0,
    actor_model=actor, critic_model=critic,gamma=0.95)
    #device=args["DEVICE"], gamma=0.95)
    


print(agent.get_action(torch.Tensor([0.5, 0.5, 3, 10, 10])))
agent.load()
print(agent.get_action(torch.Tensor([0.5, 0.5, 50, 0, 1000])))

state = env.reset()
done = False
total = 0

agent.start_episode()
state, init_reward, __ = env.step((0.5, 0.5, 3.5))  # Initial random state
num_step = 0
rewards = [init_reward]
states = [state]
while not done:
    action = agent.get_action(state)

    new_state, reward, done = env.step(action)
    transition = Transition(
        reward=reward, state=state,
        action=action, target_action=action,
        next_state=new_state)
    agent.step(transition)

    total += reward
    state = new_state
    num_step += 1
    rewards.append(reward)
    states.append(state)

y_caps = np.array(env.output())

response = y_caps[:, 0]

plt.figure(1)
plt.plot(SP, label="Set Point")
plt.plot(response, label="Response")


plt.legend()
plt.xlabel("Time")
plt.ylabel("Response")
plt.savefig('trained_response.png')
plt.show()

plt.figure(2)
plt.plot(rewards)
plt.ylabel("Reward")
plt.xlabel("Time")
plt.savefig("trained_reward.png")
plt.show()

plt.figure(3)
error = SP-response
plt.plot(error)
plt.xlabel("Time")
plt.ylabel("Error")
plt.savefig("trained_error.png")
plt.show()

plt.figure(6)
error = SP-response
plt.plot(error/3000,error)
plt.xlabel("Duty ratio")
plt.ylabel("Error")
plt.savefig("errorvsDutyRatio.png")
plt.show()

plt.figure(4)
d_error = -y_caps[:, 1]
plt.plot(d_error)
plt.xlabel("Time")
plt.ylabel("Derivative of error")
plt.savefig("trained_de_t.png")
plt.show()

# Max overshoot
error_pd = -pd.Series(error)

# Max overshoot is when |error| is maximum after touching the set point first, i.e after first crossing zero
first_cross = error_pd[((error_pd.shift() <= 0) & (error_pd >= 0))].index[0]

print("Max overshoot: ", error_pd[first_cross: ].abs().max())


# Settling time: first occurence of when tolerance band is reached
# tolerance band is +-TOLERANCE_BAND percent of the target set point

TOLERANCE_BAND = 5/100 # Within 5% of the target is tolerable
abs_tolerance = TOLERANCE_BAND*SP[0]

settling_time = t[error_pd[(error_pd.abs() < abs_tolerance)].index[0]]
print("Settling time (5%): ", settling_time)

# Settling time: first occurence of when tolerance band is reached
# tolerance band is +-TOLERANCE_BAND percent of the target set point

TOLERANCE_BAND = 2/100 # Within 5% of the target is tolerable
abs_tolerance = TOLERANCE_BAND*SP[0]

settling_time = t[error_pd[(error_pd.abs() < abs_tolerance)].index[0]]
print("Settling time (2%): ", settling_time)