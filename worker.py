import numpy as np
import torch
from memory import Memory
from agent import A3C
import utils
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
):
    T_MAX = 250

    local_agent = A3C(input_shape, n_actions)
    local_icm = ICM(input_shape, n_actions)

    memory = Memory()

    shape = [input_shape[1], input_shape[2], 1]
    env = make_env(env_id, shape)

    episode, max_eps, t_steps, scores = 0, 1000, 0, []
    best_score = min(env.reward_range)

    while episode < max_eps:
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

                loss = local_agent.calculate_loss(
                    state_,
                    hx,
                    rewards,
                    values,
                    log_probs,
                    term or trunc,
                    intrinsic_reward,
                )

                optimizer.zero_grad()
                hx = hx.detach()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_agent.parameters(), 40)

                for local_param, global_param in zip(
                    local_agent.parameters(), global_agent.parameters()
                ):
                    global_param._grad = local_param.grad

                optimizer.step()
                local_agent.load_state_dict(global_agent.state_dict())

                icm_optimizer.zero_grad()
                (inv_loss + forward_loss).backward()

                for local_param, global_param in zip(
                    local_icm.parameters(), global_icm.parameters()
                ):
                    global_param._grad = local_param.grad

                icm_optimizer.step()
                local_icm.load_state_dict(global_icm.state_dict())

                memory.clear()

        episode += 1

        if name == "1":
            scores.append(score)
            avg_score = np.mean(scores[-100:])

            if avg_score > best_score and episode > 100:
                best_score = avg_score
                torch.save(global_agent.state_dict(), f"weights/{env_id}_best.pth")

            ireward = intrinsic_reward.detach().mean()
            print(
                f"Episode: {episode}, Score: {score:.2f}, Reward: {ireward:.2f}, Avg Score: {avg_score:.2f}"
            )

    if name == "1":
        torch.save(global_agent.state_dict(), f"weights/{env_id}_final.pth")

        x = [i for i in range(episode)]
        utils.plot_learning_curve(x, scores, f"metrics/{env_id}_learning_curve.png")

        global_agent.load_state_dict(torch.load(f"weights/{env_id}_best.pth"))
        utils.save_best_version(env, global_agent, f"environments/{env_id}.gif")
