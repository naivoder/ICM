import numpy as np
import torch
from memory import Memory
from agent import A3C
from utils import plot_learning_curve
from wrappers import make_env
from icm import ICM


def worker(
    name,
    input_shape,
    n_actions,
    global_agent,
    optimizer,
    global_icm,
    icm_optimizer,
    env_id,
    global_idx,
):
    T_MAX = 20

    local_agent = A3C(input_shape, n_actions)
    local_icm = ICM(input_shape, n_actions)

    memory = Memory()

    shape = [input_shape[1], input_shape[2], 1]
    env = make_env(env_id, shape)

    episode, max_steps, t_steps, scores = 0, 5e5, 0, []

    while t_steps < max_steps:
        state, _ = env.reset()
        score, ep_steps = 0, 0
        term = trunc = False
        hx = torch.zeros(1, 256)

        while not term or trunc:
            action, value, log_prob, hx = local_agent(
                torch.tensor(np.array(state), dtype=torch.float), hx
            )
            state_, reward, term, trunc, _ = env.step(action)
            memory.remember(state, action, state_, reward, value, log_prob)

            score += reward
            state = state_

            ep_steps += 1
            t_steps += 1

            if ep_steps % T_MAX == 0 or term or trunc:
                states, actions, next_states, rewards, values, log_probs = (
                    memory.sample()
                )
                intrinsic_reward, inv_loss, forward_loss = local_icm.calculate_loss(
                    states, next_states, actions
                )

                loss = local_agent.calculate_cost(
                    state_,
                    hx,
                    rewards,
                    values,
                    log_probs,
                    term or trunc,
                    intrinsic_reward,
                )

                optimizer.zero_grad()
                icm_optimizer.zero_grad()

                hx = hx.detach()

                (inv_loss + forward_loss).backward()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_agent.parameters(), 40)

                for local_param, global_param in zip(
                    local_agent.parameters(), global_agent.parameters()
                ):
                    global_param._grad = local_param.grad

                for local_param, global_param in zip(
                    local_icm.parameters(), global_icm.parameters()
                ):
                    global_param._grad = local_param.grad

                optimizer.step()
                icm_optimizer.step()

                local_agent.load_state_dict(global_agent.state_dict())
                local_icm.load_state_dict(global_icm.state_dict())

                memory.clear()

        episode += 1
        with global_idx.get_lock():
            global_idx.value += 1

        if name == "1":
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            avg_score_5k = np.mean(scores[max(0, episode - 5000) : episode + 1])

            print(
                f"Episode: {episode}, Score: {score:.2f}, Avg Score (100) (5k): {avg_score:.2f} {avg_score_5k:.2f}"
            )

    if name == "1":
        x = [i for i in range(episode)]
        plot_learning_curve(x, scores, f"results/{env_id}_icm.png")