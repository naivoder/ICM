import matplotlib.pyplot as plt
import numpy as np
import imageio
import pandas as pd
import torch


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(x, running_avg)
    plt.title("Running Average of Previous 100 Scores")
    plt.savefig(figure_file)


def save_results(env_name, metrics, agent):
    save_prefix = env_name.split("/")[-1]
    df = pd.DataFrame(metrics)
    df.to_csv(f"csv/{save_prefix}_metrics.csv", index=False)
    plot_metrics(save_prefix, df)
    save_best_version(env_name, agent)


def plot_metrics(env, metrics):
    # episodes = np.array(metrics["episode"])
    run_avg_scores = np.array(metrics["average_score"])
    avg_values = np.array(metrics["average_critic_value"])
    episodes = np.arange(len(run_avg_scores))

    run_avg_vals = np.zeros_like(avg_values)
    for i in range(len(avg_values)):
        run_avg_vals[i] = np.mean(avg_values[max(0, i - 100) : i + 1])

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Score", color="tab:blue")
    ax1.plot(episodes, run_avg_scores, label="Average Score", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Average Critic Value", color="tab:red")
    ax2.plot(episodes, run_avg_vals, label="Average Critic Value", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plt.title(f"Average Score vs Average Critic Value per Episode in {env}")
    plt.grid(True)
    plt.savefig(f"metrics/{env}_metrics.png")
    plt.close()


def collect_fixed_states(env, n_states=10):
    """
    Collect a set of fixed initial states for monitoring average critic value.
    This function now works with a single environment and collects states
    after taking a random number of steps (1-5) before resetting.
    """
    fixed_states = []

    for _ in range(n_states):
        state, _ = env.reset()

        # Take a random number of steps (1-5)
        steps = np.random.randint(1, 6)
        for _ in range(steps):
            action = env.action_space.sample()  # Take a random action
            next_state, _, done, _, _ = env.step(action)
            if done:
                break
            state = next_state

        fixed_states.append(state)

    return np.array(fixed_states, dtype=np.float32)


def save_best_version(env, agent, fname, seeds=100):
    best_total_reward = float("-inf")
    best_frames = None

    for _ in range(seeds):
        state, _ = env.reset()

        frames = []
        total_reward = 0

        hx = torch.zeros(1, 256)
        term, trunc = False, False
        while not term and not trunc:
            frames.append(env.render())

            state = torch.tensor(np.array(state), dtype=torch.float)
            action, _, _, hx = agent(state, hx)
            next_state, reward, term, trunc, _ = env.step(action)

            total_reward += reward
            state = next_state

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_frames = frames

    save_animation(best_frames, fname)


def save_animation(frames, filename):
    with imageio.get_writer(filename, mode="I", loop=0) as writer:
        for frame in frames:
            writer.append_data(frame)
