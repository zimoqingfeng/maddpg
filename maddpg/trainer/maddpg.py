# -*- coding: utf-8 -*-
import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer

# Gamma折扣累积奖赏计算
def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r  # Recurrent discount by Gamma
        r = r*(1.-done)       # Direct discount by dones matrix
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2 # 0.99
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var)) # var_target << polyak * var_target + (1.0-polyak) * var
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

# Policy network
def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample() #act_pd.mode() #
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

# Q network
def q_train(make_obs_ph_n: object, act_space_n: object, q_index: object, q_func: object, optimizer: object, grad_norm_clipping: object = None,
            local_q_func: object = False,
            scope: object = "trainer",
            reuse: object = None,
            num_units: object = 64) -> object:
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg                                                                 # From chapter 4.2: inferring policies of other policies ???

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)         # [Important]使用 optimizer 来降低 loss, 其中变量表在 q_func_vars 中,
                                                                                                      # 保证每个变量的梯度到 grad_norm_clipping -- 梯度剪切
        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name               # name of the agent
        self.n = len(obs_shape_n)      # number of agents
        self.agent_index = agent_index # Index of the specific agent
        self.args = args               # Settings of hyper-parameters
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get()) # Creates a placeholder for a batch of tensors of a given shape and dtype.

        # [Create all the functions necessary to train the model]
        # train:             U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        # update_target_q:   make_update_exp(q_func_vars, target_q_func_vars)
        # q_values:          U.function(obs_ph_n + act_ph_n, q)
        # target_q_values:   U.function(obs_ph_n + act_ph_n, target_q)
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,                                             # String: "agent_1" or "agent_2" or ...
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,                                     # action_space.
            q_index=agent_index,                                         # Index of the specific agent.
            q_func=model,                                                # Defined model.
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),     # 优化方法 --- 自适应矩估计 --- Adam法 --- 学习率设定
            grad_norm_clipping=0.5,                                      # 梯度剪切 --- 防止梯度爆炸 --- 梯度超过该值,直接设定为该值
            local_q_func=local_q_func,
            num_units=args.num_units                                     # Hidden layers 隐藏节点数
        )
        
        # act:                U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        # train:              U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        # update_target_p:    make_update_exp(p_func_vars, target_p_func_vars)
        # p_values:           U.function([obs_ph_n[p_index]], p)
        # target_act:         U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)
        self.act, self.p_train, self.p_update, self.p_debug = p_train(   
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    # Input:  agents -->    all the trainers
    #         t      -->    increment global step counter
    # Output: loss   -->   [loss of q_train,
    #                       loss of p_train,
    #                       mean of target_q,
    #                       mean of reward,
    #                       mean of next target_q,
    #                       std of target_q]
    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        # Random sample from the replay buffer (Experience replay mechanism)
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)   # Random sample from the replay_buffer --- return the sample index.
        # collect replay sample from all agents
        obs_n = []            # Clearly, 'n' indicates the number of the total agents. (Clear the past memory.)
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):                                                # Fetch the [all agents'] information.
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)  # Fetch the observation, action, rewerds, next observation, done from the buffer.
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)                                                  # obs_n, obs_next_n, act_n
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index) # Fetch the [self] information.

        # train q network [Critic network]
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):            # obs_n + act_n + [target_q] ==> q network --> Input is all obversation and all action and the target_q --> Output is value.
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)] # 根据observation生成个体下一步的action
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))          # 根据observation以及action是计算评价
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next                            # rewards + gamma*target_q_next*(1-done) 迷之done...
        target_q /= num_sample                 # calculate the mean of the target_q
        
        q_loss = self.q_train(*(obs_n + act_n + [target_q])) # Training procedure.

        # train p network [Actor network]
        p_loss = self.p_train(*(obs_n + act_n)) # obs_n + act_n ==> list 拼接; Policy network -->  Input is all obversation and all action --> Output is action.

        self.p_update()                         # p_network: make_update_experence
        self.q_update()                         # q_network: make_update_experence

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]