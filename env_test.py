from env.pushT import PushTEnv
import numpy as np
import time


# main.py
if __name__ == "__main__":
    # 0. create env object
    env = PushTEnv()

    # 1. seed env for initial state.
    # Seed 0-200 are used for the demonstration dataset.
    env.seed(1000)

    # 2. must reset before use
    obs, _ = env.reset()

    # 3. 2D positional action space [0,512]
    action = env.action_space.sample()
    print ("Action: ", env.action_space)

    # 4. For loop to simulate the environment
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        env.render(mode='human')
        time.sleep(1)
        print ("Observation: ", obs)
        print ("Reward: ", reward)
        print ("Done: ", done)
        print ("Info: ", info)
        if done:
            break