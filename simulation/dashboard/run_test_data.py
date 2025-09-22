from __future__ import annotations

import glob
import json
import os
import re
import subprocess
from dataclasses import dataclass

import tyro
import yaml
from loguru import logger as log
from rich.logging import RichHandler

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@dataclass
class Args:
    rerun: bool = False
    # tasks: list[str] | None = None
    # task_groups: list[str] | None = None
    # robots: list[str] | None = None
    # sims: list[str] | None = None
    # commands: list[str] | None = None


args = tyro.cli(Args)


def single(command, log_dir, result_dir):
    log.info(f'Running command: "{command}"')
    with open(os.path.join(log_dir, "command.sh"), "w") as f:
        f.write(command)
    result = subprocess.run(
        command,
        shell=True,
        stdout=open(os.path.join(log_dir, "stdout.log"), "w"),
        stderr=open(os.path.join(log_dir, "stderr.log"), "w"),
        text=True,
        check=False,
    )
    if result.returncode == 0:
        log.info("Command executed successfully")
    else:
        log.error(f"Command failed with return code {result.returncode}")
    return result.returncode


def should_run(log_dir):
    if args.run_all:
        return True
    if args.run_unfinished and (
        not os.path.exists(os.path.join(log_dir, "status.txt"))
        or open(os.path.join(log_dir, "status.txt")).read() == "unfinished"
    ):
        return True
    if (
        args.run_failed
        and os.path.exists(os.path.join(log_dir, "status.txt"))
        and open(os.path.join(log_dir, "status.txt")).read() != "0"
    ):
        return True
    return False


def main():
    ROOT = "data_isaaclab/demo"

    base_log_dir = "dashboard/logs_dataset"
    base_log_path = "dashboard/logs_dataset/results.json"
    base_log_path_new = "dashboard/logs_dataset/results_new.json"

    if args.rerun:
        results = {}

        total_traj = 0
        total_frame = 0

        for dir_ in sorted(os.listdir(ROOT)):
            if not os.path.isdir(os.path.join(ROOT, dir_)):
                continue
            files = glob.glob(f"{ROOT}/{dir_}/*/demo_*/metadata.json")
            frames = glob.glob(f"{ROOT}/{dir_}/*/demo_*/*.png")
            total_traj += len(files)
            total_frame += len(frames)
            results[dir_] = {
                "total_traj": len(files),
                "total_frame": len(frames),
            }
    else:
        with open(base_log_path) as f:
            results = json.load(f)

    os.makedirs(base_log_dir, exist_ok=True)
    with open(base_log_path, "w") as f:
        json.dump(results, f)

    with open("dashboard/conf_dataset.yml") as f:
        conf = yaml.safe_load(f)
    results_new = {}
    for bench in conf["tasks"]:
        results_new[bench] = {}
        for task in conf["tasks"][bench]:
            task_ori = task
            task = task.split(":")[-1]
            total_traj = 0
            total_frame = 0
            for task_ in results.keys():
                task_name = task_.split("_")[0].split("-")[0]
                task_name = re.sub(r"(?<!^)(?=[A-Z])", "_", task_name).lower()
                if task_name != task:
                    continue
                total_traj += results[task_]["total_traj"]
                total_frame += results[task_]["total_frame"]
            results_new[bench][task_ori] = {
                "total_traj": total_traj,
                "total_frame": total_frame,
            }
    with open(base_log_path_new, "w") as f:
        json.dump(results_new, f)


if __name__ == "__main__":
    main()
