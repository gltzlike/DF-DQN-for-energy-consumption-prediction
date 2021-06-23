import numpy as np


class Prediction:

    def train(self, method_str, method, state, action, max_episodes, max_steps, file_log, state_kinds=None):

        episode_total_reward = []
        episode_step_reward = []
        episode_mae = []
        action_true = []
        action_predict = []

        for episode in range(max_episodes):

            episode_reward = 0
            count_step = 0

            index = np.random.choice(range(len(state)))
            s = state[index]

            if method_str == "DF-DQN":
                kind = state_kinds[index]

            for step in range(max_steps):
                count_step += 1

                a_true = action[index]
                action_true.append(a_true)

                if method_str == "DQN":
                    a, a_value = method.choose_action(state=s, stage="train")
                    action_predict.append(a_value)
                    r = -abs(a_value - a_true)

                elif method_str == "DF-DQN":
                    a, a_value = method.choose_action(state=s, kind=kind, stage="train")
                    action_predict.append(a_value)
                    r = -abs(a_value - a_true)

                elif method_str == "DDPG":
                    a = method.choose_action(state=s, stage="train")
                    a = np.reshape(a, (1, -1))[0][0]
                    action_predict.append(a)
                    r = -abs(a - a_true)

                episode_step_reward.append(r)

                episode_reward += r

                index += 1

                if index == len(state):
                    break

                s_ = state[index]

                if method_str == "DF-DQN":
                    kind = state_kinds[index]

                method.store_transition(s, a, r, s_)
                method.learn(count_step)

                if (index == len(state) - 1) or (step == max_steps - 1):
                    episode_total_reward.append(episode_reward)  # 保存每个回合的累计奖赏
                    print('Episode %d : %.2f' % (episode, episode_reward))
                    file_log.write('Episode %d : %.2f\n' % (episode, episode_reward))  # 打印回合数和奖赏累计值
                    break

                s = s_
            episode_reward = np.reshape(episode_reward, (1, -1))[0][0]
            episode_mae.append((-episode_reward) / count_step)  # 计算每回合的mae，看收敛情况
        return method, episode_mae, action_predict, action_true, episode_step_reward  # 将训练好的agent返回出去

    def prediction(self, method_str, method, state, action, state_kinds=None):

        action_predict = []
        action_true = []

        for i in range(len(state)):

            s = state[i]

            if method_str == "DF-DQN":
                kind = state_kinds[i]

            action_true.append(action[i])

            if method_str == "DQN":
                a, a_value = method.choose_action(state=s, stage="test")
                action_predict.append(a_value)

            elif method_str == "DF-DQN":
                a, a_value = method.choose_action(state=s, kind=kind, stage="test")
                action_predict.append(a_value)

            elif method_str == "DDPG":
                a = method.choose_action(state=s, stage="test")
                a = np.reshape(a, (1, -1))[0][0]
                action_predict.append(a)
        return action_predict, action_true
