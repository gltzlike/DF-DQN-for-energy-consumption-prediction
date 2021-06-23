from collections import deque
import random
import numpy as np
import tensorflow as tf
import tensorlayer as tl


class DDPG:

    def __init__(self, n_features, action_low, action_high, n_hidden, learning_rate_actor,
                 learning_rate_critic, gamma, tau, var, clip_min, clip_max, memory_size, batch_size):

        self.n_features = n_features
        self.action_low = action_low
        self.action_high = action_high
        self.n_hidden = n_hidden
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.gamma = gamma
        self.tau = tau
        self.var = var
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=self.memory_size)  # 构建经验池，实质上就是一个双端队列

        self.W = tf.random_normal_initializer(mean=0, stddev=0.3)
        self.b = tf.constant_initializer(0.1)

        self.actor = self.buildActor([None, self.n_features])  # 建立行动者网络
        self.actor.train()

        self.actor_target = self.buildActor([None, self.n_features])  # 创建目标行动者Q网络
        self.actor_target.eval()

        self.copyPara(self.actor, self.actor_target)  # 后面采用软更新的方式，所以此处需要先复制一遍网络参数

        self.critic = self.buildCritic([None, self.n_features], [None, 1])  # Critic网络需要s和a的值，a是一维的
        self.critic.train()

        self.critic_target = self.buildCritic([None, self.n_features], [None, 1])
        self.critic_target.eval()

        self.copyPara(self.critic, self.critic_target)

        self.actor_opt = tf.optimizers.Adam(self.learning_rate_actor)
        self.critic_opt = tf.optimizers.Adam(self.learning_rate_critic)

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)  # soft replacement

    def buildActor(self, inputs_shape):

        x = tl.layers.Input(inputs_shape)

        hidden_first = tl.layers.Dense(n_units=self.n_hidden, act=tf.nn.relu,
                                       W_init=tf.initializers.GlorotUniform(), b_init=self.b)(x)
        hidden_second = tl.layers.Dense(n_units=self.n_hidden, act=tf.nn.relu,
                                        W_init=tf.initializers.GlorotUniform(), b_init=self.b)(hidden_first)
        output = tl.layers.Dense(n_units=1, W_init=tf.initializers.GlorotUniform(), b_init=self.b)(hidden_second)

        return tl.models.Model(inputs=x, outputs=output)

    def buildCritic(self, inputs_state_shape, inputs_action_shape):

        s = tl.layers.Input(inputs_state_shape)
        a = tl.layers.Input(inputs_action_shape)
        x = tl.layers.Concat(1)([s, a])

        hidden_first = tl.layers.Dense(n_units=self.n_hidden, act=tf.nn.relu,
                                       W_init=self.W, b_init=self.b)(x)
        hidden_second = tl.layers.Dense(n_units=self.n_hidden,
                                        W_init=self.W, b_init=self.b)(hidden_first)
        y = tl.layers.Dense(n_units=1, W_init=self.W, b_init=self.b)(hidden_second)

        return tl.models.Model(inputs=x, outputs=y)

    def copyPara(self, from_model, to_model):

        for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
            j.assign(i)

    def emaUpdate(self):

        paras = self.actor.trainable_weights + self.critic.trainable_weights

        self.ema.apply(paras)

        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))

    def updateNetwork(self, state, action, reward, next_state):

        with tf.GradientTape() as tape:
            next_action = self.actor_target(next_state)

            q_next = self.critic_target([next_state, next_action])

            q_target = reward + self.gamma * q_next

            q = self.critic([state, action])

            td_error = tf.losses.mean_absolute_error(q_target, q)

        critic_grads = tape.gradient(td_error, self.critic.trainable_weights)

        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

        with tf.GradientTape(persistent=True) as tape:
            action_pre = self.actor(state)

            q_pre = self.critic([state, action_pre])

            actor_loss = -tf.reduce_mean(q_pre)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)

        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))

    def choose_action(self, state, stage):

        state = np.reshape(state, (1, -1))
        state = np.array(state, dtype="float32")

        action = self.actor(state)

        if stage == "train":

            action = np.random.normal(action, self.var)
        return action

    def store_transition(self, state, action, reward, next_state):

        state = np.reshape(state, (1, -1))
        next_state = np.reshape(next_state, (1, -1))
        action = np.reshape(action, (1, -1))
        reward = np.reshape(reward, (1, -1))

        transition = np.concatenate((state, action, reward, next_state), axis=1)
        self.memory.append(transition[0])

    def learn(self, step):

        if len(self.memory) == self.memory_size:

            if step % 200 == 0:
                self.emaUpdate()
                self.var = self.var * 0.995

            batch = np.array(random.sample(self.memory, self.batch_size), dtype="float32")
            batch_s = batch[:, :self.n_features]
            batch_a = batch[:, self.n_features:(self.n_features + 1)]
            batch_r = batch[:, (self.n_features + 1):(self.n_features + 2)]
            batch_s_ = batch[:, (self.n_features + 2):(self.n_features * 2 + 2)]

            self.updateNetwork(state=batch_s, action=batch_a, reward=batch_r, next_state=batch_s_)
