from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lbc_husky
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import math
import time
import torch
import argparse
import datetime

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."
# weight_path = home_path + "/raisimGymTorch/data/husky_navigation/2021-11-15-11-10-34/full_4200.pt"
# weight_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-11-26-10-33-49/full_4600.pt" #6.8s
# weight_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-01-21-16-24/full_4600.pt" #6.476500s
# weight_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-01-22-16-11/full_5000.pt" #6.5605s
# weight_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-01-23-26-13/full_1800.pt" #1800-6.305000
# weight_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-02-00-33-04/full_4000.pt" #4000-6.64s
# weight_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-02-01-33-08/full_3800.pt" #3800-6.4330
# weight_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-07-18-07-04/full_2400.pt" #4800 - 6.1625s
# weight_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-08-14-29-40 6s AdjustedScanWidth OriGoalVelHeight/full_5000.pt" # 6s"
# weight_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-09-16-09-45 Redo 6s/full_5000.pt" #6.7s
# weight_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-11-12-30-31 6.09s/full_5000.pt" #5000-6.09s
# weight_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-11-13-39-43 torqueCur2000/full_4400.pt" #4400-6.15s
# weight_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-11-14-40-16  /full_5000.pt" #6.13
weight_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-11-12-30-31 6.09s/full_5000.pt" #5000-6.09s
dir_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-11-20-06-49"
dir_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-11-20-06-49 increaseDepth_decreaseLIdar" #1024
# dir_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-11-22-32-51 addheight+tuninghyper"
# dir_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-12-00-58-31 5.5s"
# dir_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-12-02-05-17" #4000-5.6s
# dir_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-12-12-12-26" #5.8s
dir_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-12-14-46-06"

# dir_path = "/home/dhammiko/raisim/LearningBasedControl_FinalProject2021/raisimGymTorch/data/husky_navigation/2021-12-26-05-39-42"

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
cfg['environment']['num_envs'] = 200
env = VecEnv(lbc_husky.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_0-5000" +'.mp4')
for iterate in range(0,5001,200):
    weight_path = dir_path + "/full_" + str(iterate) + ".pt"
    # shortcuts
    ob_dim = env.num_obs
    act_dim = env.num_acts

    iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
    weight_dir = weight_path.rsplit('/', 1)[0] + '/'
    if weight_path == "":
        print("Can't find trained weight, please provide a trained weight with --weight switch\n")
    else:
        print("Loaded weight from {}\n".format(weight_path))
        start = time.time()
        env.reset()
        completion_sum = 0
        done_sum = 0
        average_dones = 0.
        n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
        total_steps = n_steps * 1
        start_step_id = 0

        print("Visualizing and evaluating the policy: ", weight_path)
        loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim, act_dim)
        loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

        env.load_scaling(weight_dir, int(iteration_number))

        # env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str("NICHA")+'.mp4')
        env.turn_on_visualization()
        max_steps = 80 ## 8 secs 80
        completed_sum = 0

        for step in range(max_steps):
            # time.sleep(cfg['environment']['control_dt'])
            obs = env.observe(False)
            action = loaded_graph.architecture(torch.from_numpy(obs).cpu())
            reward, dones, not_completed = env.step(action.cpu().detach().numpy())
            completed_sum = completed_sum + sum(not_completed)

        print('----------------------------------------------------')
        print('{:<40} {:>6}'.format("avg completion time: ", '{:0.6f}'.format(completed_sum / env.num_envs * cfg['environment']['control_dt'])))
        print('----------------------------------------------------\n')

        env.turn_off_visualization()
env.stop_video_recording()