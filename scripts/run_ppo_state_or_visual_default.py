import argparse

import gymnasium as gym
import mani_skill.envs

from mani_skill.utils.wrappers import RecordEpisode

import subprocess as sp
import time

def pick_cube_intro():
    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="state", # there is also "state_dict", "rgbd", ...
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array"
    )

    env = RecordEpisode(
        env,
        "/home/filip-grigorov/code/scripts/data/videos", # the directory to save replay videos and trajectories to
        # on GPU sim we record intervals, not by single episodes as there are multiple envs
        # each 100 steps a new video is saved
        max_steps_per_video=100
    )

    print(f"Observational space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, _ = env.reset(seed=0)
    done = False

    while not done:
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        env.render()

    env.close()




def run_default_state_ppo(task, name):
    print("Running state-based PPO")

    start_time = time.time()

    sp.run([
        "python", 
        "examples/baselines/ppo/ppo.py", 
        f"--env_id={task}", 
        f"--exp-name={name}",
        f"--num_envs={40}", #1024 default
        f"--update_epochs={8}", 
        f"--num_minibatches={16}", #32
        f"--total_timesteps={1_000_000}", #600_000
        f"--eval_freq={8}",
        f"--num-steps={20}"
    ])

    end_time = time.time()
    elapsed_sec = (end_time - start_time)
    print(f"Elasped (seconds): {elapsed_sec}")

    print("End")


def run_default_visual_ppo(task, name, render_quality):
    print("Running visual-based PPO")

    start_time = time.time()

    sp.run([
        "python", 
        "examples/baselines/ppo/ppo_rgb_custom.py", 
        f"--env_id={task}", 
        f"--exp-name={name}",
        f"--num_envs={40}", #256 default
        f"--update_epochs={8}", 
        f"--num_minibatches={16}",
        f"--total_timesteps={1_000_000}",
        f"--eval_freq={10}",
        f"--num-steps={20}",
        f"--sim_quality={render_quality}" # rasterization
    ])

    end_time = time.time()
    elapsed_sec = (end_time - start_time)
    print(f"Elasped (seconds): {elapsed_sec}")

    print("End")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action='store_true', help="Just state") # default is False
    parser.add_argument("--state", action='store_true', help="Just visual") # default is False

    args = parser.parse_args()

    if args.visual:
        tasks = [
            #"PushCube-v1", 
            #"PickCube-v1", 
            #"StackCube-v1", 
            #"PegInsertionSide-v1", 
            #"AssemblingKits-v1", # realsense
            #"PlugCharger-v1"
        ]

        RENDER_QUALITY = "high"
        MODALITY = "rgb"
        EXPERIMENT = "iterations/runs-5M" # modality or sim-quality etc.
        names = [
            #f"{EXPERIMENT}-{MODALITY}-pushcube-{RENDER_QUALITY}", 
            #f"{EXPERIMENT}-{MODALITY}-pickcube-{RENDER_QUALITY}",
            #f"{EXPERIMENT}-{MODALITY}-stackcube-{RENDER_QUALITY}",
            #f"{EXPERIMENT}-{MODALITY}-peginsertionside-{RENDER_QUALITY}",
            #f"{EXPERIMENT}-{MODALITY}-assemblingkits-{RENDER_QUALITY}",
            #f"{EXPERIMENT}-{MODALITY}-plugcharger-{RENDER_QUALITY}", 
        ]

        assert len(tasks) == len(names), "equal number of params"

        print(f"Render quality is {RENDER_QUALITY}")

        for idx in range(len(tasks)):
            task_name = tasks[idx]
            name = names[idx]
            print(f"Running experiment for {task_name}-{name}")
            run_default_visual_ppo(task=task_name, name=name, render_quality=RENDER_QUALITY)
            wait_time_s = 3
            print(f"Waiting for {wait_time_s} seconds ...")
            time.sleep(wait_time_s)

    elif args.state:
        tasks = [
            #"PushCube-v1", 
            #"PickCube-v1", 
            #"StackCube-v1", 
            #"PegInsertionSide-v1", 
            #"AssemblingKits-v1", # realsense
            #"PlugCharger-v1"
        ]

        RENDER_QUALITY = ""
        MODALITY = "state"
        EXPERIMENT = "iterations/runs-5M" # modality or sim-quality etc.
        names = [
            #f"{EXPERIMENT}-{MODALITY}-pushcube-{RENDER_QUALITY}",
            #f"{EXPERIMENT}-{MODALITY}-pickcube-{RENDER_QUALITY}",
            #f"{EXPERIMENT}-{MODALITY}-stackcube-{RENDER_QUALITY}",
            #f"{EXPERIMENT}-{MODALITY}-peginsertionside-{RENDER_QUALITY}",
            #f"{EXPERIMENT}-{MODALITY}-assemblingkits-{RENDER_QUALITY}",
            #f"{EXPERIMENT}-{MODALITY}-plugcharger-{RENDER_QUALITY}", 
        ]

        for idx in range(len(tasks)):
            task_name = tasks[idx]
            name = names[idx]
            print(f"Running experiment for {task_name}-{name}")
            run_default_state_ppo(task=task_name, name=name)
            wait_time_s = 3
            print(f"Waiting for {wait_time_s} seconds ...")
            time.sleep(wait_time_s)

    else:
        print("No option has been selected")
