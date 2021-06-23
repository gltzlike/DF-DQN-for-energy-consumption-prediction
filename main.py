import os
import time
import shutil
import numpy as np
import pandas as pd
from tool import Tool
from dqn_agent import DQN
from ddpg_agent import DDPG
from prediction import Prediction
from draw_picture import DrawPicture
from create_samples import CreateSample
from state_classify import StateClassify
from df_dqn_agent import DF_DQN
from data_preprocessing import DataPreprocessing

"""
    1. data_preprocessing
"""
dir = os.getcwd()
dir_data = dir + "/data/"

filename_data = "index_0_Q (2015.1.1-2016.12.31).csv"

# 1.1 outliers are detected and replaced
if not (os.path.exists(dir_data + filename_data)):
    filename_raw_data = "6 train.csv"
    building_id = "0"
    energy_type = "Q"
    filename_Q = "index_0_Q.csv"

    filename_outliers = "index_0_Q (2015.1.1-2016.10.31).csv"
    n_neighbors = 48

    DataPreprocessing().segQW(load_filename=filename_raw_data, build_filename=filename_Q,
                              building_id=building_id, energy_type=energy_type)

    DataPreprocessing().outliersProcess(load_filename=filename_Q, build_filename=filename_outliers,
                                        n_neighbors=n_neighbors)

    DataPreprocessing().joint(left_filename=filename_outliers, right_filename=filename_Q, build_filename=filename_data)

# 1.2 features extraction + construct samples and labels
# load data (2015/1/1-2016/12/31)
data = pd.read_csv(dir_data + filename_data, header=None)
data = np.array(data[1])
data = np.reshape(data, (-1, 1))

features = 24
data, label = CreateSample().createSample(data=data, shape="vector", features=features)

data_train = np.array(data[0:len(data) - 61 * 24])
label_train = np.array(label[0:len(label) - 61 * 24])

data_test = np.array(data[len(data) - 61 * 24:])
label_test = np.array(label[len(label) - 61 * 24:])

# 1.3 normalization
filename_log = ""
data_train_scale, data_test_scale = Tool(filename_log).normalization(data_train=data_train, data_test=data_test)

"""
    2. DRL methods for energy consumption prediction
"""
MAX_EPISODES = 5
MAX_STEPS = 1000
CLASS = 1
N_CLASS = 3
iteration = 1
data_train_min = np.min(data_train)
data_train_max = np.max(data_train)

METHOD_STR = "DF-DQN"  # DQN，DF-DQN，DDPG

dir_dqn = dir + '\\experiments\\DQN' + '\\DQN_index='
dir_ddpg = dir + '\\experiments\\DDPG' + '\\DDPG_index='

while True:

    CLASS = CLASS + 1

    dir_dfdqn = dir + '\\experiments\\DF-DQN' + '\\N_Class=' + str(CLASS) + '\\DF-DQN_index='

    for index in np.arange(0, iteration):

        start_first = time.perf_counter()

        # determine the path of methods
        if METHOD_STR == "DF-DQN":
            dir_choose = dir_dfdqn + str(index)

        elif METHOD_STR == "DQN":
            dir_choose = dir_dqn + str(index)

        elif METHOD_STR == "DDPG":
            dir_choose = dir_ddpg + str(index)

        if os.path.exists(dir_choose):
            shutil.rmtree(dir_choose)
        os.makedirs(dir_choose)

        filename_log = "log.txt"
        file_log = open(dir_choose + "\\" + filename_log, 'w')

        # 2.1 DF-DQN
        if METHOD_STR == "DF-DQN":

            gap = np.ceil((data_train_max - data_train_min + 1) / CLASS)  # size of sub-action space

            print("\nclass: ", str(CLASS), " iterations: ", str(index))
            print("the number of actions: ", str(gap))

            file_log.write("class: " + str(CLASS) + " iterations: " + str(index) + "\n")
            file_log.write("the number of actions: " + str(gap) + " \n\n")

            class_train_true = ((label_train - data_train_min) / gap).astype(int)
            class_test_true = ((label_test - data_train_min) / gap).astype(int)

            state_train_scale, state_test_scale, class_train_pre, class_test_pre = StateClassify().constructState(
                data_train_scale=data_train_scale, data_test_scale=data_test_scale,
                class_train_true=class_train_true, class_test_true=class_test_true, file_log=file_log)

            start_second = time.perf_counter()

            # hyper-parameters
            N_FEATURES = features + CLASS
            ACTION_START = data_train_min
            ACTION_END = data_train_min + gap
            N_ACTIONS = int(gap)
            N_HIDDEN = 32
            LEARNING_RATE = 0.01
            GAMMA = 0.9
            EPSILON = 0.5
            EPSILON_DECAY = 0.995
            EPSILON_MIN = 0.01
            MEMORY_SIZE = 2000
            BATCH_SIZE = 64

            df_dqn = DF_DQN(n_features=N_FEATURES, n_class=CLASS, action_start=ACTION_START,
                            action_end=ACTION_END,
                            n_actions=N_ACTIONS, n_hidden=N_HIDDEN, learning_rate=LEARNING_RATE, gamma=GAMMA,
                            epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN,
                            memory_size=MEMORY_SIZE,
                            batch_size=BATCH_SIZE)

            dqn_train, mae_train, predict_train, actual_train, reward_train = Prediction(). \
                train(method_str=METHOD_STR, method=df_dqn, state=state_train_scale, action=label_train,
                      max_episodes=MAX_EPISODES, max_steps=MAX_STEPS, file_log=file_log, state_kinds=class_train_pre)

            predict_test, actual_test = Prediction().prediction(
                method_str="DF-DQN", method=dqn_train, state=state_test_scale, action=label_test,
                state_kinds=class_test_pre)

        elif METHOD_STR == "DQN":

            print("iterations: ", str(index))
            file_log.write("iterations: " + str(index) + "\n")

            N_FEATURES = features
            N_ACTIONS = int(data_train_max - data_train_min + 1)
            ACTION_LOW = data_train_min
            ACTION_HIGH = data_train_max
            N_HIDDEN = 32
            LEARNING_RATE = 0.01
            GAMMA = 0.9
            EPSILON = 0.5
            EPSILON_DECAY = 0.995
            EPSILON_MIN = 0.01
            MEMORY_SIZE = 2000
            BATCH_SIZE = 64

            dqn = DQN(n_features=N_FEATURES, n_actions=N_ACTIONS, n_hidden=N_HIDDEN, action_low=ACTION_LOW,
                      action_high=ACTION_HIGH,
                      learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON, epsilon_decay=EPSILON_DECAY,
                      epsilon_min=EPSILON_MIN, memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE)

            dqn_train, mae_train, predict_train, actual_train, reward_train = Prediction().train(
                method_str=METHOD_STR, method=dqn, state=data_train_scale,
                action=label_train, max_episodes=MAX_EPISODES, max_steps=MAX_STEPS, file_log=file_log)

            predict_test, actual_test = Prediction().prediction(
                method_str="DQN", method=dqn_train, state=data_test_scale, action=label_test)

        elif METHOD_STR == "DDPG":

            print("iterations: ", str(index))
            file_log.write("iterations: " + str(index) + "\n")

            N_FEATURES = features
            ACTION_LOW = data_train_min
            ACTION_HIGH = data_train_max
            CLIP_MIN = data_train_min
            CLIP_MAX = data_train_max
            N_HIDDEN = 32
            LEARNING_RATE_ACTOR = 0.001
            LEARNING_RATE_CRITIC = 0.001
            GAMMA = 0.9
            TAU = 0.1
            VAR = 40
            MEMORY_SIZE = 2000
            BATCH_SIZE = 64

            ddpg = DDPG(n_features=N_FEATURES, action_low=ACTION_LOW, action_high=ACTION_HIGH, n_hidden=N_HIDDEN,
                        learning_rate_actor=LEARNING_RATE_ACTOR, learning_rate_critic=LEARNING_RATE_CRITIC,
                        gamma=GAMMA, tau=TAU, var=VAR, clip_min=CLIP_MIN, clip_max=CLIP_MAX, memory_size=MEMORY_SIZE,
                        batch_size=BATCH_SIZE)

            ddpg_train, mae_train, predict_train, actual_train, reward_train = Prediction().train(
                method_str=METHOD_STR, method=ddpg, state=data_train, action=label_train,
                max_episodes=MAX_EPISODES, max_steps=MAX_STEPS, file_log=file_log)

            predict_test, actual_test = Prediction().prediction(
                method_str=METHOD_STR, method=ddpg_train, state=data_test, action=label_test)


        # save data
        save_data_list = [mae_train, predict_train, actual_train, reward_train, predict_test, actual_test]
        save_data_filename = ["mae_train.csv", "predict_train.csv", "actual_train.csv", "reward_train.csv",
                              "predict_test.csv", "actual_test.csv"]

        for j in range(len(save_data_list)):

            data_temp = pd.DataFrame(save_data_list[j])

            data_temp_filename = "\\" + save_data_filename[j]

            if os.path.exists(dir_choose + data_temp_filename):
                os.remove(dir_choose + data_temp_filename)

            data_temp.to_csv(dir_choose + data_temp_filename, index=False, header=None)

        # prediction accuracy
        print("training set: ")
        file_log.write("\ntraining set: \n")
        Tool(file_log).mae(action_predict=predict_train, action_true=actual_train)
        Tool(file_log).rmse(action_predict=predict_train, action_true=actual_train)
        Tool(file_log).cv(action_predict=predict_train, action_true=actual_train)

        print("====================================================================")
        print("test set: ")
        file_log.write("====================================================================\n")
        file_log.write("test set: \n")
        Tool(file_log).mae(action_predict=predict_test, action_true=actual_test)
        Tool(file_log).rmse(action_predict=predict_test, action_true=actual_test)
        Tool(file_log).cv(action_predict=predict_test, action_true=actual_test)

        # plot
        DrawPicture().Xrange_Y(dir=dir_choose, figName="mae_train", Yname="mae", Y=mae_train)

        DrawPicture().Xrange_Ypredicted_Yactual(dir=dir_choose, figName="predict and actual in test set",
                                                action_predict=predict_test, action_true=actual_test)

        # 关闭日志
        file_log.close()

    if METHOD_STR == "DQN" or METHOD_STR == "DDPG":
        break
    elif METHOD_STR == "DF-DQN":
        if CLASS == N_CLASS:
            break
