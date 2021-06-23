from collections import deque
import random
import tensorflow as tf
import tensorlayer as tl
import numpy as np


class DF_DQN:
    def __init__(self, n_features, n_class, action_start, action_end, n_actions, n_hidden,
                 learning_rate, gamma, epsilon, epsilon_decay, epsilon_min, memory_size, batch_size):

        self.n_features = n_features
        self.n_class = n_class
        self.action_start = action_start
        self.action_end = action_end
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=self.memory_size)  # 构建经验池，实质上就是一个双端队列

        self.W = tf.random_normal_initializer(mean=0, stddev=0.3)
        self.b = tf.constant_initializer(0.1)

        # action space is divided equally
        self.action_space = []
        temp = []
        for i in range(self.n_class):
            temp = np.linspace(self.action_start + i * (self.action_end - self.action_start),
                               self.action_start + (i + 1) * (self.action_end - self.action_start),
                               self.n_actions, endpoint=False)
            self.action_space.append(temp)

        self.Q_network = self.buildTensorModel([None, self.n_features])
        self.Q_network.train()
        self.target_Q_network = self.buildTensorModel([None, self.n_features])
        self.target_Q_network.eval()

        self.opt = tf.optimizers.Adam(self.learning_rate)

    def buildTensorModel(self, inputs_shape):

        x = tl.layers.Input(inputs_shape)
        h1 = tl.layers.Dense(n_units=self.n_hidden, act=tf.nn.relu)(x)
        h2 = tl.layers.Dense(n_units=self.n_hidden, act=tf.nn.relu)(h1)
        y = tl.layers.Dense(n_units=self.n_actions, act=None)(h2)
        network = tl.models.Model(inputs=x, outputs=y)
        return network

    def updateTensorTargetQ(self):

        for i, target in zip(self.Q_network.trainable_weights, self.target_Q_network.trainable_weights):
            target.assign(i)

    def updateTensorQ(self, state, action, reward, next_state):

        q = self.Q_network(np.array(state, dtype='float32')).numpy()

        q_next = self.target_Q_network(np.array(next_state, dtype='float32')).numpy()
        q_next_max = np.reshape(np.max(q_next, axis=1), (-1, 1))
        q_target = reward + self.gamma * q_next_max

        for row in range(0, len(action)):
            q[row][int(action[row][0])] = q_target[row][0]

        with tf.GradientTape() as tape:
            q_raw = self.Q_network(np.array(state, dtype='float32'))
            loss = tl.cost.mean_squared_error(q, q_raw)

        grads = tape.gradient(loss, self.Q_network.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.Q_network.trainable_weights))

    def updateEpsilon(self):

        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def choose_action(self, state, stage, kind):

        if stage == "train":
            if np.random.rand() >= self.epsilon:
                q = self.Q_network(
                    np.array(state, dtype='float32').reshape([-1, self.n_features]))
                a_index = np.argmax(q)

            else:
                a_index = random.randint(0, self.n_actions - 1)

        else:
            q = self.Q_network(np.array(state, dtype='float32').reshape([-1, self.n_features]))
            a_index = np.argmax(q)

        a_value = self.action_space[kind][a_index]
        return a_index, a_value

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
                self.updateEpsilon()
                self.updateTensorTargetQ()

            batch = np.array(random.sample(self.memory, self.batch_size))
            batch_s = batch[:, :self.n_features]
            batch_a = batch[:, self.n_features:(self.n_features + 1)]
            batch_r = batch[:, (self.n_features + 1):(self.n_features + 2)]
            batch_s_ = batch[:, (self.n_features + 2):(self.n_features * 2 + 2)]

            self.updateTensorQ(state=batch_s, action=batch_a, reward=batch_r, next_state=batch_s_)
