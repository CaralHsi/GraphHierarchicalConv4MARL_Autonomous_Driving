import numpy as np
import random
import tensorflow as tf
import graphconv.common.tf_util as U
import numpy.matlib as matlib

from graphconv.common.distributions import make_pdtype
from graphconv import AgentTrainer
from graphconv.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def p_train(name, make_obs_ph_n, adj_n, act_space_n, neighbor_n, p_index, p_func, q_func, num_adversaries, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=128, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        agent_n = len(obs_ph_n)
        vec_n = U.BatchInput([1, neighbor_n], name="vec").get()

        p_input1 = obs_ph_n[0: num_adversaries] if name == "adversaries" else obs_ph_n[num_adversaries: agent_n]
        p_input2 = adj_n[0: num_adversaries] if name == "adversaries" else adj_n[num_adversaries: agent_n]
        p_input3 = vec_n

        # call for actor network
        # act_space is not good!!!!!!!!!!
        p = p_func(p_input1, p_input2, p_input3, neighbor_n, num_adversaries if name == "adversaries" else (agent_n - num_adversaries),
                   5, scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = []
        act_sample = []
        for i in range(0, num_adversaries) if name == "adversaries" else range(num_adversaries, agent_n):
            act_pd_temp = act_pdtype_n[i].pdfromflat(p[i - (0 if name == "adversaries" else num_adversaries)])
            act_pd.append(act_pd_temp)
            act_sample.append(act_pd_temp.sample())

        temp = []
        for i in range(len(act_pd)):
            temp.append(act_pd[i].flatparam())

        # Is this regularization method correct?????????????????????????????/
        p_reg = tf.reduce_mean(tf.square(temp))

        act_input_n = act_ph_n + []
        for i in range(0, num_adversaries) if name == "adversaries" else range(num_adversaries, agent_n):
            act_input_n[i] = act_sample[i - (0 if name == "adversaries" else num_adversaries)]

        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + adj_n + [vec_n], outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=p_input1 + (adj_n[0: num_adversaries] if name == "adversaries" else adj_n[num_adversaries: agent_n]) + [p_input3], outputs=act_sample, list_output=True)
        p_values = U.function(p_input1 + (adj_n[0: num_adversaries] if name == "adversaries" else adj_n[num_adversaries: agent_n]) + [p_input3], p, list_output=True)

        # target network
        target_p = p_func(p_input1, p_input2, p_input3, neighbor_n, num_adversaries if name == "adversaries" else (
                agent_n - num_adversaries), 5, scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = []
        for i in range(0, num_adversaries) if name == "adversaries" else range(num_adversaries, agent_n):
            target_act_sample.append(act_pdtype_n[i].pdfromflat(target_p[i - (0 if name == "adversaries" else num_adversaries)]).sample())
        target_act = U.function(inputs=p_input1 + (adj_n[0: num_adversaries] if name == "adversaries" else adj_n[num_adversaries: agent_n]) + [p_input3], outputs=target_act_sample, list_output=True)

        return act, train, update_target_p, p_values, target_act


def q_train(make_obs_ph_n, adj_n, act_space_n, neighbor_n, q_index, q_func, agent_n, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:, 0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss
        # + 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:, 0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, q_values, target_q_values


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, p_model, q_model, obs_shape_n, act_space_n, num_adversaries, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.args = args
        self.neighbor_n = 2
        self.num_adversaries = num_adversaries
        adj_n = []
        obs_ph_n = []
        agent_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())
            adj_n.append(U.BatchInput([self.neighbor_n,
                                       num_adversaries if i < num_adversaries else (self.n - num_adversaries)],
                                      name="adjacency" + str(i)).get())
        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_values, self.target_q_values = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            adj_n=adj_n,
            act_space_n=act_space_n,
            neighbor_n=self.neighbor_n,
            q_index=self.n,
            q_func=q_model,
            agent_n=self.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
        )

        self.act, self.p_train, self.p_update, self.p_values, self.target_act = p_train(
            name=self.name,
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            adj_n=adj_n,
            act_space_n=act_space_n,
            neighbor_n=self.neighbor_n,
            p_index=agent_n,
            p_func=p_model,
            q_func=q_model,
            num_adversaries=self.num_adversaries,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
        )

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        for _ in range(len(obs)):
            obs[_] = obs[_][None]
        return self.act(*obs)

    def experience(self, obs, act, rew, new_obs, done, adj, new_adj, terminal):
        # Store transition in the replay buffer.
        done_int = [float(x) for x in done]
        self.replay_buffer.add(obs, act, rew, new_obs, done_int, adj, new_adj)

    def pre_update(self):
        self.replay_sample_index = None

    def update(self, agents, t, restore=False):
        if not restore:
            if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
                return
        elif len(self.replay_buffer) < self.args.batch_size:
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        index = self.replay_sample_index
        obs_n = []
        obs_next_n = []
        act_n = []
        adj_n = []
        adj_next_n = []
        for i in range(len(agents)):
            obs_record, act_record, rew_record, obs_next_record, done_record, adj_record, adj_next_record = \
                agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs_record)
            obs_next_n.append(obs_next_record)
            act_n.append(act_record)
            adj_n.append(adj_record)
            adj_next_n.append(adj_next_record)

        obs, act, rew, obs_next, done, adj, adj_next = self.replay_buffer.sample_index(index)

        target_act_next_n = []
        target_q_next_input_obs = []
        target_q_next_input_act = []
        q_input_obs_n = []
        q_input_act_n = []
        p_input_adj_n = []

        for _, agent in enumerate(agents):  # traverse every species
            q_input_obs = []
            q_input_act = []
            p_input_adj = []
            target_act_next_input_obs = []
            target_act_next_input_adj = []
            for j in range(obs_n[_].shape[1]):  # traverse every agent in each species
                _obs = []
                _act = []
                _adj = []
                _obs_next = []
                _adj_next = []
                for i in range(self.args.batch_size):  # traverse each instance
                    _obs.append(obs_n[_][i][j])
                    _act.append(act_n[_][i][j])
                    _adj.append(adj_n[_][i][j])
                    _obs_next.append(obs_next_n[_][i][j])
                    _adj_next.append(adj_next_n[_][i][j])
                q_input_obs.append(np.array(_obs))
                q_input_act.append(np.array(_act))
                p_input_adj.append(np.array(_adj))
                target_act_next_input_obs.append(np.array(_obs_next))
                target_act_next_input_adj.append(np.array(_adj_next))
            vec = matlib.repmat([1, 0], self.args.batch_size, 1)
            vec = np.expand_dims(vec, axis=1)
            target_act_next_input = target_act_next_input_obs + target_act_next_input_adj + [vec]
            temp = agent.target_act(*target_act_next_input)
            target_act_next_n.append(temp)
            target_q_next_input_obs.extend(target_act_next_input_obs)
            target_q_next_input_act.extend(temp)
            q_input_obs_n.extend(q_input_obs)
            q_input_act_n.extend(q_input_act)
            p_input_adj_n.extend(p_input_adj)

        target_q = 0.0
        target_q_next = self.target_q_values(*(target_q_next_input_obs + target_q_next_input_act))
        rew = np.sum(rew, 1)/4
        # used to be (1 - done) but actually what's 'done' is not defined in "simple-world-comm" scenario,
        # thus should be considered again how to define "done" for species
        target_q += rew + self.args.gamma * target_q_next

        # train the critic network
        q_train_input = q_input_obs_n + q_input_act_n + [target_q]
        q_loss = self.q_train(*q_train_input)

        # train the policy network
        p_loss = self.p_train(*(q_input_obs_n + q_input_act_n + p_input_adj_n + [vec]))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
