import os
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from worker import worker
from agent import A3C
from icm import ICM
from ale_py import ALEInterface, LoggerMode
from config import environments
import warnings
import gymnasium as gym

warnings.simplefilter("ignore")
ALEInterface.setLoggerMode(LoggerMode.Error)


class ParallelEnv:
    def __init__(self, env_id, n_threads, input_shape, n_actions, n_games):
        names = [str(i) for i in range(n_threads)]

        global_agent = A3C(input_shape, n_actions)
        global_agent.share_memory()  # ???

        global_icm = ICM(input_shape, n_actions)
        global_icm.share_memory()  # ???

        optimizer = SharedAdam(global_agent.parameters(), lr=3e-5)
        icm_optimizer = SharedAdam(global_icm.parameters(), lr=3e-5)

        self.ps = [
            mp.Process(
                target=worker,
                args=(
                    name,
                    input_shape,
                    n_actions,
                    global_agent,
                    optimizer,
                    global_icm,
                    icm_optimizer,
                    env_id,
                    n_games,
                ),
            )
            for name in names
        ]

        [p.start() for p in self.ps]
        [p.join() for p in self.ps]


if __name__ == "__main__":
    from argparse import ArgumentParser
    import torch

    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--env", default=None, help="Environment name from Gymnasium"
    )
    parser.add_argument(
        "--n_threads",
        default=4,
        type=int,
        help="Number of parallel environments during training",
    )
    parser.add_argument(
        "--n_games",
        default=5000,
        type=int,
        help="Total number of episodes (games) to play during training",
    )
    args = parser.parse_args()

    # os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(4)


    for fname in ["metrics", "environments", "weights", "csv"]:
        if not os.path.exists(fname):
            os.makedirs(fname)

    input_shape = (4, 42, 42)

    mp.set_start_method("spawn")
    if args.env:
        config_env = gym.make(args.env)
        n_actions = config_env.action_space.n

        print("Environment:", args.env)
        print("Observation space:", config_env.observation_space)
        print("Action space:", config_env.action_space)

        env = ParallelEnv(
            args.env, args.n_threads, input_shape, n_actions, args.n_games
        )
    else:
        for env_name in environments:
            args.env = env_name
            config_env = gym.make(args.env)
            n_actions = config_env.action_space.n

            print("Environment:", args.env)
            print("Observation space:", config_env.observation_space)
            print("Action space:", config_env.action_space)

            env = ParallelEnv(
                args.env, args.n_threads, input_shape, n_actions, args.n_games
            )
