#  MIT License
#
#  Copyright (c) 2017 Ilya Kostrikov
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import argparse
import sys

import numpy as np
import torch

import gym
import my_pybullet_envs
import random
import pickle

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.arguments import parse_args_with_unknown
from a2c_ppo_acktr import utils


sys.path.append("a2c_ppo_acktr")

parser = argparse.ArgumentParser(description="RL")
parser.add_argument(
    "--seed", type=int, default=1, help="random seed (default: 1)"
)
parser.add_argument(
    "--env-name",
    default="HumanoidSwimmerEnv-v1",
    help="environment to load and test on",
)
parser.add_argument(
    "--src-env-name",
    default="",
    help="environment to transfer policy from ("" if same as test env)",
)
parser.add_argument(
    "--load-dir",
    default="./trained_models/",
    help="directory to save agent logs (default: ./trained_models/)",
)
parser.add_argument(
    "--non-det",
    type=int,
    default=0,
    help="whether to use a non-deterministic policy, 1 true 0 false",
)
parser.add_argument(
    "--iter",
    type=int,
    default=None,
    help="which iter pi to test"
)
parser.add_argument(
    '--cuda',
    action='store_true',
    default=False,
    help='use cuda during testing'
)
parser.add_argument(
    "--num-trajs",
    type=int,
    default=200,
    help="how many trajs to rollout for testing",
)

args, extra_dict = parse_args_with_unknown(parser)

np.set_printoptions(precision=2, suppress=None, threshold=sys.maxsize)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

is_cuda = args.cuda
device = "cuda" if is_cuda else "cpu"

args.det = not args.non_det

# If render is provided, use that. Otherwise, turn it on.
if "render" not in extra_dict:
    extra_dict["render"] = True

env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device=device,
    allow_early_resets=False,
    **extra_dict,
)
# dont know why there are so many wrappers in make_vec_envs...
env_core = env.venv.venv.envs[0].env.env

if args.src_env_name == "":
    env_name_transfer = args.env_name
else:
    env_name_transfer = args.src_env_name
actor_critic, ob_rms, recurrent_hidden_states, masks \
    = utils.load(args.load_dir, env_name_transfer, is_cuda, args.iter)

if ob_rms:
    print(ob_rms.mean)
    print(ob_rms.var)
    print(ob_rms.count)

vec_norm = utils.get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

obs = env.reset()
# print("obs", obs)
# input("reset, press enter")
done = False

reward_total = 0
list_rewards = []
list_traj_lengths = []
list_r_per_step = []
dist = None
timer = 0
cur_traj_idx = 0

while True:

    try:
        env_core.cam_track_torso_link()
        dist = env_core.get_dist()
    except:
        pass

    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det
        )

    obs, reward, done, info = env.step(action)
    timer += 1


    reward_np = reward.cpu().numpy()[0][0]
    reward_total += reward_np
    list_r_per_step.append(reward_np)

    if done:
        list_rewards.append(reward_total)
        list_traj_lengths.append(len(list_r_per_step))
        print(
            f"{args.load_dir}\t"
            f"tr: {reward_total:.1f}\t"
            # f"x: {dist:.2f}\t"
            f"tr_per_step_r_ave: {reward_total/len(list_r_per_step):.2f}\t"
            f"total_per_step_r_ave: {np.sum(list_rewards)/np.sum(list_traj_lengths):.2f}\t"
        )
        reward_total = 0.0
        timer = 0

        cur_traj_idx += 1
        if cur_traj_idx >= args.num_trajs:
            break

        list_r_per_step = []

    masks.fill_(0.0 if done else 1.0)

print("mean episode reward,", np.sum(list_rewards)/args.num_trajs)
print("median episode reward,", np.median(list_rewards))
print("mean episode len", np.sum(list_traj_lengths)/args.num_trajs)
# bins_list = np.arange(60) * 100.0
# print(bins_list)
# plt.hist(list_rewards, alpha=0.5, label='r hist', bins=bins_list)
# plt.legend(loc='upper right')
# plt.show()
