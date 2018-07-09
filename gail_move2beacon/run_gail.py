from policy_net import Policy_net
from discriminator import Discriminator
from ppo import PPOTrain
import sys
from absl import flags
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
import matplotlib.pyplot as plt
from action_group import actAgent2Pysc2, no_operation
from state_group import obs2state, obs2distance
import numpy as np
import random
import tensorflow as tf
from collections import deque
import time
import math
# Define the constant
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED_UNIT = features.SCREEN_FEATURES.selected.index
friendly = 1
neutral = 3
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NO_OP           = actions.FUNCTIONS.no_op.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_SELECT_ALL  = [0]
_NOT_QUEUED  = [0]
step_mul = 4
FLAGS = flags.FLAGS
EPISODES = 10000
BATCH_SIZE = 500

# main function, create env, define model, learn from observation and save model
def train():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(
                        map_name='MoveToBeacon',
                        agent_interface_format=sc2_env.parse_agent_interface_format(
                            feature_screen=64,
                            feature_minimap=64,
                            rgb_screen=None,
                            rgb_minimap=None,
                            action_space=None,
                            use_feature_units=False),
                        step_mul=step_mul,
                        game_steps_per_episode=None,
                        disable_fog=False,
                    visualize=False) as env:
        r = tf.placeholder(tf.float32)  ########
        rr = tf.summary.scalar('reward', r)
        merged = tf.summary.merge_all() ########
        expert_observations = np.genfromtxt('trajectory/observations.csv')
        expert_actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)
        with tf.Session() as sess:
            Policy = Policy_net('policy', 2, 4)
            Old_Policy = Policy_net('old_policy', 2, 4)
            PPO = PPOTrain(Policy, Old_Policy)
            D = Discriminator()
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            writer = tf.summary.FileWriter('./board/gail', sess.graph) ########
            c = 0
            for episodes in range(100000):
                done = False
                obs = env.reset()
                while not 331 in obs[0].observation.available_actions:
                    actions = actAgent2Pysc2(100, obs)
                    obs = env.step(actions=[actions])
                state = obs2state(obs)
                observations = []
                actions_list = []
                rewards = []
                v_preds = []
                reward = 0
                global_step = 0
                while not done:
                    global_step += 1
                    state = np.stack([state]).astype(dtype=np.float32)
                    act, v_pred = Policy.act(obs=state, stochastic=True)
                    act, v_pred = np.asscalar(act), np.asscalar(v_pred)
                    observations.append(state)
                    actions_list.append(act)
                    rewards.append(reward)
                    v_preds.append(v_pred)
                    actions = actAgent2Pysc2(act, obs)
                    obs = env.step(actions=[actions])
                    next_state = obs2state(obs)
                    distance = obs2distance(obs)
                    if distance < 0.03 or global_step == 100:
                        done = True
                    if done:
                        v_preds_next = v_preds[1:] + [0]
                        break
                    state = next_state
                observations = np.reshape(observations, newshape=[-1, 2])
                actions_list = np.array(actions_list).astype(dtype=np.int32)
                for i in range(2):
                    sample_indices = (np.random.randint(expert_observations.shape[0], size=observations.shape[0]))
                    inp = [expert_observations, expert_actions]
                    sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp] # sample training data
                    D.train(expert_s=sampled_inp[0],
                            expert_a=sampled_inp[1],
                            agent_s=observations,
                            agent_a=actions_list)
                d_rewards = D.get_rewards(agent_s=observations, agent_a=actions_list)
                d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)

                gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)               
                gaes = np.array(gaes).astype(dtype=np.float32)
                v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

                inp = [observations, actions_list, gaes, d_rewards, v_preds_next]
                PPO.assign_policy_parameters()
                for epoch in range(15):
                    sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                    size=32)  # indices are in [low, high)
                    sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                    PPO.train(obs=sampled_inp[0],
                            actions=sampled_inp[1],
                            gaes=sampled_inp[2],
                            rewards=sampled_inp[3],
                            v_preds_next=sampled_inp[4])
                summary = sess.run(merged, feed_dict={r: global_step})
                writer.add_summary(summary, episodes)
                if global_step < 50:
                    c += 1
                else:
                    c = 0
                if c > 10:
                    saver.save(sess, './model/gail.cpkt')
                    print('save model')
                    break
                print(episodes, global_step, c)

if __name__ == '__main__':
    train()