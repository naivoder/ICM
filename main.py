import os
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from worker import worker
from agent import A3C
from icm import ICM

os.environ["OMP_NUM_THREADS"] = "1"


class ParallelEnv:
    def __init__(self, env_id, n_threads, input_shape, n_actions, global_ep):
        names = [str(i) for i in range(n_threads)]

        global_agent = A3C(input_shape, n_actions)
        global_agent.share_memory()  # ???

        global_icm = ICM(input_shape, n_actions)
        global_icm.share_memory()  # ???

        optimizer = SharedAdam(global_agent.parameters(), lr=1e-4)
        icm_optimizer = SharedAdam(global_icm.parameters(), lr=1e-4)

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
                    global_ep,
                ),
            )
            for name in names
        ]

        [p.start() for p in self.ps]
        [p.join() for p in self.ps]


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    gloabl_ep = mp.Value("i", 0)

    env_id = "SpaceInvadersNoFrameskip-v4"
    n_threads = 4
    n_actions = 6  # need to change this to be dynamic
    input_shape = (4, 42, 42)

    env = ParallelEnv(env_id, n_threads, input_shape, n_actions, gloabl_ep)
