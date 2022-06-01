from tkinter.filedialog import SaveFileDialog
import numpy as np
import pandas as pd

import torch

from matplotlib import pyplot as plt

from collections import deque
import pickle
from pid import PIDModel
from agent import Agent, Actor, Critic, Transition

save_states = []
def train(args):
    T_SIZE = 100
    SET_POINT = 100

    t = np.linspace(0, 1000, num=T_SIZE)
    #SP = np.ones(T_SIZE)*SET_POINT
    #Step-Adding through set points(of velocity)
    SP = np.ones(len(t))*1000
    #SP[0: 5] = 0
    #SP[5: 25] = 500
    #SP[26: 50] = 1500
    #SP[51:75 ] = 1750
    #SP[76:99 ] = 2000
    env = PIDModel(ku=1.396, tu=3.28, t=t, SP=SP)

    actor = Actor()
    critic = Critic()
    agent = Agent(env,
        actor_lr=args["ACTOR_LEARNING_RATE"], critic_lr=args["CRITIC_LEARNING_RATE"],
        actor_model=actor, critic_model=critic,
        device=args["DEVICE"], gamma=args["GAMMA"])

    stats = {
        "episode_reward": deque([]),
        "del_ts": []
    }

    if args["LOAD_PREVIOUS"]:
        print("Loading previously trained model")
        agent.load()

    for i in range(args["NUM_EPISODES"]):
        print("Starting episode", i)
        state = env.reset()
        total = 0

        agent.start_episode()
        state, _, __ = env.step((0.5, 0.5, 3.5))  # Initial random state

        num_step = 0; done = False
        while not done:
            action = agent.get_action(state)

            # Exploration strategy
            gauss_noise = np.random.normal(0, args["exploration_stddev"], size=3)
            target_action = action+torch.Tensor(gauss_noise)
            target_action = agent.actor_model.clamp_action(target_action)

            new_state, reward, done = env.step(target_action.detach().numpy())

            transition = Transition(
                    reward=reward, state=state,
                    action=action, target_action=target_action,
                    next_state=new_state)
            agent.step(transition)

            if (num_step % args["PRINT_EVERY"] == 0):
                print("\tStep", num_step, "for episode", i)
                print("\t", action, target_action)
                print("\tReward accumulated:", total)

            assert (type(target_action) == torch.Tensor)
            assert (target_action.requires_grad)
            assert (action.requires_grad)

            total += reward
            state = new_state
            num_step += 1
            save_states.append(new_state)

        # Learn from this episode
        agent.learn()

        if i%1==0:
            agent.save()
            stats["episode_reward"].append(total/num_step)
            stats["del_ts"].extend(agent.get_episode_stats()[1])

            print("Reward is ", total, "and average reward is", total/num_step)

    return stats


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    stats = train({
        "NUM_EPISODES": 50,
        "DEVICE": "cpu",
        "exploration_stddev": 0.1,
        "LOAD_PREVIOUS": False,
        "PRINT_EVERY": 50,
        "GAMMA": 0.95,
        "CRITIC_LEARNING_RATE": 1e-5,
        "ACTOR_LEARNING_RATE": 1e-4
    })



    with open('saved_gains.pkl', 'wb') as f:
       pickle.dump(save_states, f)
    plt.plot(stats["episode_reward"])
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    #plt.yticks([0 -0.2 -0.4 -0.6 -0.8 -1.0 -1.2 -1.4 -1.6])
    plt.savefig("average_reward_per_episode_graph.png")
    plt.show()

    del_ts = pd.Series(stats["del_ts"])
    del_ts.to_csv("del_ts.csv")
    plt.plot(del_ts)
    plt.savefig("del_ts.png")
    plt.show()


