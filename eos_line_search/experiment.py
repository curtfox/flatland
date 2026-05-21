from dataclasses import dataclass
from eos_line_search.run import *
import numpy as np
import os
import random


@dataclass
class Experiment:
    runs: list
    device: str
    path: str

    def run_experiment(self, use_wb=False, entity=None, project_name=None, group=None):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)

        experiment_directory = os.path.join(self.path, "experiments")

        ### Training for each run
        for run_num, run in enumerate(self.runs):
            print("-----Run " + str(run_num + 1) + "-----")
            run.perform_run(
                use_wb,
                entity,
                project_name,
                group,
                experiment_directory,
                self.device,
                seed,
            )

        print("Done")
