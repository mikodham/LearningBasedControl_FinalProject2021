from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lbc_husky
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse


# task specification
task_name = "husky_navigation"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE: ", device)
if device == torch.device('cpu'): exit(3)
# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))  # load configuration file

# create environment from the configuration file
env = VecEnv(lbc_husky.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
# instantiates an environment and it uses this lbc huskey environment, see line 2 from raisimGymTorch.env.bin import lbc_husky
# see lbc_husky.cpython-...gnu,so, load this compiled library, which is created when you run the setup, and run the code.

# shortcuts
ob_dim = env.num_obs # get # of observations and actions
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_iteration_steps = n_steps * env.num_envs
total_steps = 0

avg_rewards = []
#  find in cgg.yaml to pass the parameter cfg['architecture']['policy_net']
actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
                         # pass input size =ob_dim, output_size = activation_dimension,
                         # go to algo/ppo/module MLP module MLP(shape, actionvation_fn, input_size, output_size):
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),  # in ppomodule, dim and std
                         device) # pass the distribution, because actor is stochastic function
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                           device) # critic is deterministic

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"]) # saves in data, husky_navigation
tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.999,  # discount factor initially 0.996
              lam=0.95,  # lambda for GAE
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              entropy_coef=0.005,  # ADD
              )

if mode == 'retrain':
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

for update in range(1000000): # START of actual training
    start = time.time()  # just for measuring how much time it takes for the training
    env.reset() # resets the environment, send the robot to where it should start
    reward_sum = 0
    done_sum = 0
    completed_sum = 0
    average_dones = 0.

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        print("Total steps so far: ", total_steps)
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        # if update=200,400,..., every 200l, save the policy, value, network parameters and save ppo.
        # save in full_update.pt file in data directory
        # we create another graph just to demonstrate the save/load method
        loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
        loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])
        # actor is already a graph, so no need to define loaded graph again actually, to demonstrate how to load the graph

        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        for step in range(n_steps):  # this part is just visualization to see how actor performed and recording, not training
            frame_start = time.time()
            obs = env.observe(False)  # get environment, pass it to loaded actor,
            action = loaded_graph.architecture(torch.from_numpy(obs).cpu()) # then input action to the environment
            reward, dones, completed = env.step(action.cpu().detach().numpy())
            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)

        env.stop_video_recording()
        env.turn_off_visualization()

        env.reset()
        env.save_scaling(saver.data_dir, str(update))

    # actual training
    for step in range(n_steps):  # collect batch of data, offline training
        obs = env.observe()
        action = ppo.observe(obs)  # ppo to evaluate action
        reward, dones, not_completed = env.step(action) # collect big batch of data
        ppo.step(value_obs=obs, rews=reward, dones=dones)
        # let ppo know the reward, dones, not completed, to improve the agent
        done_sum = done_sum + sum(dones)  # the following is for loggings
        reward_sum = reward_sum + sum(reward)
        completed_sum = completed_sum + sum(not_completed)

    # data constraints - DO NOT CHANGE THIS BLOCK
    total_steps += env.num_envs * n_steps
    if total_steps > 20000000:
        break

    # take st step to get value obs
    obs = env.observe()  # get last observatoion
    ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update) # use batch of data to imrpove policy
    average_performance = reward_sum / total_iteration_steps
    average_dones = done_sum / total_iteration_steps
    avg_rewards.append(average_performance)
    # if you train in simulations, you can actually define different observations for actor and critique
    # here actor_obs and value_obs use the same observation, in practice you can use different observations,
    # that's why we have 2 different argument
    # log this iteration (for TF board) only if update is multiple of 10

    # curriculum update. Implement it in Environment.hpp
    env.curriculum_callback()

    end = time.time()

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("avg reward: ", '{:0.10f}'.format(average_performance)))
    print('{:<40} {:>6}'.format("avg completion time: ", '{:0.6f}'.format(completed_sum / env.num_envs * cfg['environment']['control_dt'])))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_iteration_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_iteration_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('std: ')
    print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')
