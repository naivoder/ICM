import gymnasium as gym
import numpy as np
from collections import deque
import cv2


class RepeatAction(gym.Wrapper):
    def __init__(self, env, repeat=4, fire_first=False):
        super(RepeatAction, self).__init__(env)
        self.repeat = repeat
        self.fire_first = fire_first
        self.shape = env.observation_space.low.shape

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            state, reward, term, trunc, info = self.env.step(action)
            t_reward += reward

            done = term or trunc
            if done:
                break

        return state, t_reward, term, trunc, info

    def reset(self, seed=None, options=None):
        state, _ = self.env.reset(seed=seed, options=options)
        if self.fire_first:
            if self.env.unwrapped.get_action_meanings()[1] == "FIRE":
                state, _, _, _, _ = self.env.step(1)
        return state, {}


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(0.0, 1.0, self.shape, dtype=np.float32)

    def observation(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, self.shape[1:], interpolation=cv2.INTER_AREA)
        return state / 255.0


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat=4):
        super(StackFrames, self).__init__(env)
        self.repeat = int(repeat)
        self.stack = deque([], maxlen=self.repeat)

        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(repeat, axis=0),
            env.observation_space.high.repeat(repeat, axis=0),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        self.stack.clear()
        state, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.stack.maxlen):
            self.stack.append(state)
        return np.array(self.stack).reshape(self.observation_space.low.shape), info

    def observation(self, state):
        self.stack.append(state)
        return np.array(self.stack).reshape(self.observation_space.low.shape)


def make_env(env_name, shape=(42, 42, 1), repeat=4):
    env = gym.make(env_name, render_mode="rgb_array")
    env = RepeatAction(env)
    env = PreprocessFrame(env, shape)
    env = StackFrames(env, repeat)
    return env


if __name__ == "__main__":
    env = make_env("SpaceInvadersNoFrameskip-v4")
    state, _ = env.reset()

    print("Expected Shape:", env.observation_space.low.shape)
    print("Actual Shape:", state.shape)
