# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
'''
parkour policy:
    obs:(num_envs,num_obs) (4,753)
    action:(num_envs,joint) (4,12)

avoid_policy:
    obs: (num_envs,num_obs) (1,61)
    action:(1,12)
'''

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer
sys.path.append("/Share/ziyanli/extreme-parkour")
from predictor.train import MLPRewardPredictor  # Import the predictor model
import numpy as np  # Add this import if not already present
import multiprocessing as mp

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def create_recording_camera(gym, env_handle,
        resolution= (1920, 1080),
        h_fov= 86,
        actor_to_attach= None,
        transform= None, # related to actor_to_attach
    ):
    camera_props = gymapi.CameraProperties()
    camera_props.enable_tensors = True
    camera_props.width = resolution[0]
    camera_props.height = resolution[1]
    camera_props.horizontal_fov = h_fov
    camera_handle = gym.create_camera_sensor(env_handle, camera_props)
    if actor_to_attach is not None:
        gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            actor_to_attach,
            transform,
            gymapi.FOLLOW_POSITION,
        )
    elif transform is not None:
        gym.set_camera_transform(
            camera_handle,
            env_handle,
            transform,
        )
    return camera_handle


def play(args, flag):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    
    # safe_path = '/Share/ziyanli/ABS/training/legged_gym/logs/a1_pos_rough_lag/12_06_12-50-07_/'

    # safe_path = "/Share/ziyanli/ABS/training/legged_gym/logs/a1_pos_rough_lag/12_15_22-22-04_"
    safe_path = "/Share/ziyanli/ABS/training/legged_gym/logs/a1_pos_rough_lag/12_02_23-32-57_/"
    # safe_path = "/Share/ziyanli/ABS/training/legged_gym/logs/a1_pos_rough_lag/12_06_12-50-07_/"
    
    # safe_path = '/Share/ziyanli/ABS/training/legged_gym/logs/a1_pos_rough_lag/12_06_12-50-07_/'

    # safe_path = "/Share/ziyanli/ABS/training/legged_gym/logs/a1_pos_rough_lag/12_15_22-22-04_"
    # safe_path = "/Share/ziyanli/ABS/training/legged_gym/logs/a1_pos_rough_lag/12_02_23-32-57_/"
    # safe_path = "/Share/ziyanli/ABS/training/legged_gym/logs/a1_pos_rough_lag/12_06_12-50-07_/"
    # log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid
    log_pth = "/Share/ziyanli/extreme-parkour/legged_gym/logs/parkour_new/dis-10k-1113/"

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg_safe, train_cfg_safe = task_registry.get_cfgs(name=args.task_safe)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 4 if not args.save else 64
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "rough stairs up": 0., 
                                    "rough stairs down": 0., 
                                    "discrete": 0., 
                                    "stepping stones": 0.0,
                                    "gaps": 0., 
                                    "smooth flat": 0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "large stairs up": 0.,
                                    "large stairs down": 0.,
                                    "parkour": 0.,
                                    "parkour_hurdle": 0.6,
                                    "parkour_flat": 0.,
                                    "parkour_step": 0.,
                                    "parkour_gap": 0.2, 
                                    "demo": 0.2}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True

    env_cfg.terrain.selected = True

    # env_cfg.terrain.selected_idx = 18
    # env_cfg.terrain.terrain_kwargs = {
    #     "type": "parkour_step_terrain",
    #     "num_stones": env_cfg.terrain.num_goals - 2,
    #     "step_height": 0.35,
    #     # "step_height": 0.55,
    #     "x_range": [0.3,1.5],
    #     "y_range": [-0.4, 0.4],
    #     "half_valid_width": [0.5, 1],
    #     "pad_height": 0,
    # }
    # env_cfg.terrain.selected_idx = 15
    # difficulty = 0.1
    # x_range = [-0.1, 0.1+0.3*difficulty]  # offset to stone_len
    # y_range = [0.2, 0.3+0.1*difficulty]
    # stone_len = [0.9 - 0.3*difficulty, 1 - 0.2*difficulty]#2 * round((0.6) / 2.0, 1)
    # incline_height = 0.25*difficulty
    # last_incline_height = incline_height + 0.1 - 0.1*difficulty
    # env_cfg.terrain.terrain_kwargs = {
    #     "type": "parkour_terrain",
    #     "num_stones": env_cfg.terrain.num_goals - 2,
    #     "x_range": x_range,
    #     "y_range": y_range,
    #     "incline_height": incline_height,
    #     "stone_len": stone_len,
    #     "stone_width": 1.0, 
    #     "last_incline_height": last_incline_height,
    #     "pad_height": 0,
    #     "pit_depth": [0.2, 1]
    # }
    env_cfg.terrain.selected_idx = 16
    env_cfg.terrain.terrain_kwargs = {
        "type": "parkour_hurdle_terrain",
        "num_stones": env_cfg.terrain.num_goals - 2,
        "stone_len": 0.4,
        # "hurdle_height_range": [0.3, 0.6],
        # "hurdle_height_range": [0.6, 1.0],
        # "hurdle_height_range": [0.3, 0.6],
        "hurdle_height_range": [0.2, 0.5, 1.0],
        # "hurdle_height_range": [0.5, 0.6, 1.0],
        "pad_height": 0,
        # "x_range": [1.2, 2.2],
        # "x_range": [3.6, 6.6],
        # "x_range": [3.0, 5.2],
        "x_range": [4.2, 6.6],
        # "x_range": [6.0,7.0],

        # "x_range": [4.2,5.0],
        # "x_range": [6.0,7.0],

        "y_range": env_cfg.terrain.y_range,
        "half_valid_width": [0.4, 0.8],
    }
    
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.max_push_vel_xy = 0.0
    # env_cfg.domain_rand.randomize_dof_bias = False
    # env_cfg.domain_rand.erfi = False
    # env_cfg.domain_rand.randomize_timer_minus = 0.0
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.added_mass_range = [0, 0]

    # Load the trained predictor model
    past_len = 20
    predictor = MLPRewardPredictor(latent_dim=32, past_steps=past_len)
    predictor.load_state_dict(torch.load('/Share/ziyanli/extreme-parkour/predictor/ckpts/predictor_1121_20_0_100.pth', map_location=torch.device('cuda:1')))
    # predictor.load_state_dict(torch.load('/Share/ziyanli/extreme-parkour/predictor/ckpts/predictor_1121_20_0_100.pth'))
    predictor.eval()

    # Load the statistics file
    statistics = np.load('/Share/ziyanli/extreme-parkour/predictor/statistics.npy', allow_pickle=True).item()
    rewards_mean = statistics['mean']
    rewards_std = statistics['std']
    
    depth_latent_buffer = deque(maxlen=past_len)  # Buffer to store past depth latents
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # env.lookat_id = 2
    obs = env.get_observations()
    safe_obs = env.get_safe_observations()
    
    print("lookat", env.lookat_id)
    # env.terrain_levels[:] = 9

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    
    # load safe policy
    train_cfg_safe.runner.resume = True
    ppo_runner_safe, train_cfg_safe = task_registry.make_alg_runner(log_root=safe_path, env=env, name=args.task, args=args, train_cfg=train_cfg_safe)
    policy_safe = ppo_runner_safe.get_inference_policy(device=env.device)

    if RECORD_FRAMES:
        print("RECORD FRAMES")
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(*env_cfg.viewer.pos)
        transform.r = gymapi.Quat.from_euler_zyx(0., 0., -np.pi/2)
        recording_camera = create_recording_camera(
            env.gym,
            env.envs[0],
            transform= transform,
        )
        if not os.path.exists(os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "images")):
            os.makedirs(os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "images"))
        print("RECORD FRAMES11")

    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None
    predicted_reward = 0.
    REWARD_THRESHOLD = 0.015  # Set the reward threshold

    for i in range(10*int(env.max_episode_length)):
        if args.use_jit:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    depth_latent = torch.ones((env_cfg.env.num_envs, 32), device=env.device)
                    actions, depth_latent = policy_jit(obs.detach(), True, infos["depth"], depth_latent)
                else:
                    depth_buffer = torch.ones((env_cfg.env.num_envs, 58, 87), device=env.device)
                    actions, depth_latent = policy_jit(obs.detach(), False, depth_buffer, depth_latent)
            else:
                obs_jit = torch.cat((obs.detach()[:, :env_cfg.env.n_proprio+env_cfg.env.n_priv], obs.detach()[:, -env_cfg.env.history_len*env_cfg.env.n_proprio:]), dim=1)
                actions = policy(obs_jit)
        else:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                    obs_student[:, 6:8] = 0
                    depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = depth_latent_and_yaw[:, -2:]
                    depth_latent_buffer.append(depth_latent.detach().cpu().numpy())
                
                    if len(depth_latent_buffer) == past_len:
                        past_latents = np.stack(depth_latent_buffer, axis=1)  # Shape: (num_envs, past_steps, 32)
                        past_latents = torch.tensor(past_latents, dtype=torch.float32)
                        past_latents = past_latents.unsqueeze(0)  # Add batch dimension
                        with torch.no_grad():
                            predicted_rewards = predictor(past_latents).cpu().numpy()  # Convert tensor to numpy array
                            predicted_rewards = predicted_rewards.squeeze(0)  # Remove batch dimension
                        predicted_reward = predicted_rewards[env.lookat_id]
                        predicted_reward = predicted_reward * rewards_std + rewards_mean
                obs[:, 6:8] = 1.5*yaw
                    
            else:
                depth_latent = None
            
            if hasattr(ppo_runner.alg, "depth_actor"):
                actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            else:
                actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)

        actions_safe = policy_safe(safe_obs.detach())
        
        if flag.value:
        # if predicted_reward <= REWARD_THRESHOLD:
            obs, safe_obs, _, rews, dones, infos = env.step(actions_safe.detach(), is_safe=True)
        else:
            obs, safe_obs, _, rews, dones, infos = env.step(actions.detach(), is_safe=False)

        # Display rewards and predicted rewards using OpenCV
        frame = np.zeros((200, 800, 3), dtype=np.uint8)
        rewards_np = rews[env.lookat_id].cpu().numpy()  # Convert tensor to numpy array
        cv2.putText(frame, f"Rewards: {rewards_np:.3f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Predicted Rewards: {np.round(predicted_reward, 3)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if flag.value:
            cv2.circle(frame, (400, 150), 50, (0, 0, 255), -1)  # Red circle
        else:
            cv2.circle(frame, (400, 150), 50, (0, 255, 0), -1)  # Green circle
        # Display predicted reward at the top of the window
        cv2.putText(frame, f"Predicted Reward: {np.round(predicted_reward, 3)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Rewards and Predicted Rewards', frame)
        cv2.waitKey(1)

        if RECORD_REW:
            # Normalize depth values to the range 0-255
            depth_image = env.depth_buffer[env.lookat_id, -1].cpu().numpy()
            depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255
            depth_image = depth_image.astype(np.uint8)
            depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
            depth_image = cv2.resize(depth_image, (depth_image.shape[1] * 4, depth_image.shape[0] * 4))  # Resize the image to make it larger
            text = f"Rewards: {rewards_np:.2f} | step: {i}"
            text_size = 0.5
            text_thickness = 1
            text_color = (0, 0, 255)  # Red color in BGR format
            text_position = (10, depth_image.shape[0] - 10)
            # cv2.putText(depth_image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, text_thickness, cv2.LINE_AA)
            
            depth_filename = os.path.join(os.path.abspath("/Share/ziyanli/extreme-parkour/legged_gym/logs/images3/"), f"depth_{i:04d}.png")
            cv2.imwrite(depth_filename, depth_image)

        if RECORD_FRAMES:
            print("RECORD FRAMES")
            filename = os.path.join(
                os.path.abspath("../../logs/images/"),
                f"{i:04d}.png",
            )
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            print("RECORD FRAMES222")

        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
        print("time:", env.episode_length_buf[env.lookat_id].item() / 50, 
              "cmd vx", env.commands[env.lookat_id, 0].item(),
              "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )
        # exit()
        id = env.lookat_id


def listen_for_keypress(flag):
    """Process to listen for keypress and toggle the safe action flag."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.imshow("Switch window", img)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            flag.value = not flag.value
        sleep(0.01)  # Avoid CPU overuse
        

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    RECORD_REW = True
    MOVE_CAMERA = False
    args = get_args()
    use_safe_action_flag = mp.Value('b', False)  # Shared flag for safe action
    try:
        process = mp.Process(target=listen_for_keypress, args=(use_safe_action_flag,))
        process.start()
        play(args, use_safe_action_flag)
    except KeyboardInterrupt:
        process.join()
        print("KeyboardInterrupt")
        process.terminate()
    finally:
        if RECORD_FRAMES:
            import subprocess
            print("converting frames to video")
            log_dir = "../../logs/{}/".format(args.proj_name) + args.exptid
            subprocess.run(["ffmpeg",
                "-framerate", "50",
                "-r", "50",
                "-i", "../../logs/images/%04d.png",
                "-c:v", "libx264",
                "-hide_banner", "-loglevel", "error",
                os.path.join(log_dir, f"video_4.mp4")
            ])
            print("done converting frames to video, deleting frame images")
            for f in os.listdir(os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "images")):
                os.remove(os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "images", f))
            print("done deleting frame images")
        # if RECORD_REW:
            # import subprocess
            # print("converting depth images to video")
            # log_dir = "/Share/ziyanli/extreme-parkour/legged_gym/logs/{}/".format(args.proj_name) + args.exptid
            # # Convert depth images to video
            # subprocess.run(["ffmpeg",
            #     "-framerate", "50",
            #     "-r", "50",
            #     "-i", "../../logs/images/depth_%04d.png",
            #     "-c:v", "libx264",
            #     "-hide_banner", "-loglevel", "error",
            #     os.path.join(log_dir, f"depth_video.mp4")
            # ])
            # print("done converting depth images to video, deleting depth images")
            # for f in os.listdir(os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "images")):
            #     if f.startswith("depth_"):
            #         os.remove(os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "images", f))
            # print("done deleting depth images")
    # play(args)
