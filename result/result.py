# 对实验数据进行取平均
import os
import shutil
import pandas as pd
import numpy as np

dir_result = os.getcwd()  # 返回当前文件的路径
dir = os.path.abspath(os.path.dirname(dir_result)) + '\\experiments\\'  # DQN代码结果的路径

method_list = ["DQN", "DF-DQN", "DDPG"]

count_start = 0  # 计算平均值的起始样本
count_end = 10  # 计算平均值的终止样本

CLASS = 1  # DF-DQN的起始类别
N_CLASS = 36  # DF-DQN的终止类别

CLASS_ACC = []
RUN_TIME = []

DQN_RUN = []
DDPG_RUN = []

# 计算DQN，DF-DQN，DDPG预测的平均值
for method in method_list:

    while True:

        # 训练过程中保存的数据 mae_train, predict_test, actual_test
        mae_train = []
        predict_test = []
        actual_test = []

        save_data_list = [mae_train, predict_test, actual_test]
        save_data_filename = ["mae_train", "predict_test", "actual_test"]

        if method == "DF-DQN":

            CLASS = CLASS + 1
            dir_method = dir + method + "\\" + "N_CLASS=" + str(CLASS) + "\\"

        elif method == "DQN" or method == "DDPG":
            dir_method = dir + method + "\\"  # D:\（4）DF-DQN\experiments\DQN

        """
            读取每次试验的计算时间和DF-DQN的分类准确度
        """
        for index in np.arange(count_start, count_end):

            dir_temp = dir_method + method + "_index=" + str(index) + "\\"

            f = open(dir_temp + 'log.txt', 'r')

            for i, line in enumerate(f.readlines()):  # 仅适合读取小文件

                if method == "DF-DQN":
                    if i == 3:
                        line = float(line.split(" ")[1])
                        CLASS_ACC.append(line)

                    # 24类，index=3时，有个特例
                    elif CLASS == 24 and index == 3 and i == 208:
                        line = float(line.split("：")[1])
                        RUN_TIME.append(line)
                        break

                    elif i == 209:
                        line = float(line.split("：")[1])
                        RUN_TIME.append(line)
                        break

                elif method == "DDPG":
                    if i == 205:
                        line = float(line.split("：")[1])
                        DDPG_RUN.append(line)
                        break

                elif method == "DQN":
                    # index = 3 有特例
                    if index == 3 and i == 204:
                        line = float(line.split("：")[1])
                        DQN_RUN.append(line)
                        break

                    elif i == 205:
                        line = float(line.split("：")[1])
                        DQN_RUN.append(line)
                        break

            f.close()

        """
        # 将所有实验值进行平均，然后存储到新的文件中
        # 读取所有次试验的预测值
        for index in np.arange(count_start, count_end):
            
            dir_temp = dir_method + method + "_index=" + str(index) + "\\"

            # 循环读取多个文件
            for j in range(len(save_data_list)):
                dir_index = dir_temp + save_data_filename[j] + ".csv"
                save_data_list[j].append(np.array(pd.read_csv(dir_index, header=None)))

        # 判断相应的文件夹是否存在
        dir_temp_mean = dir_method + method + "_mean" + "\\"
        if os.path.exists(dir_temp_mean):
            shutil.rmtree(dir_temp_mean)
        os.makedirs(dir_temp_mean)

        # 转换为矩阵求均值，然后保存到csv文件中
        for j in range(len(save_data_list)):
            temp_mean = np.mean(np.reshape(save_data_list[j], (count_end - count_start, -1)), axis=0)
            temp_mean = pd.DataFrame(temp_mean)
            temp_mean.to_csv(dir_temp_mean + save_data_filename[j] + "_mean.csv", index=False, header=None)
        """

        if method == "DF-DQN":
            if CLASS == N_CLASS:
                break
        elif method == "DQN" or method == "DDPG":
            break

"""
    对读取到的准确率和运行时间进行平均值的处理
"""
DQN_RUN = np.reshape(DQN_RUN, (-1, 10))
DDPG_RUN = np.reshape(DDPG_RUN, (-1, 10))

CLASS_ACC = np.reshape(CLASS_ACC, (-1, 10))
RUN_TIME = np.reshape(RUN_TIME, (-1, 10))

print(np.mean(DQN_RUN, axis=1))

# 行求均值, 并记录
filename_log = "run_time.txt"
file_log = open(dir + filename_log, 'w')

file_log.write("DQN的平均计算时间：" + str(np.mean(DQN_RUN, axis=1)) + "\n")
file_log.write("DDPG的平均计算时间：" + str(np.mean(DDPG_RUN, axis=1)) + "\n\n")


for i in range(len(CLASS_ACC)):
    j = i + 2
    file_log.write(str(j) + "的类别准确率：" + str(np.mean(CLASS_ACC, axis=1)[i]))
    file_log.write("\t平均计算时间：" + str(np.mean(RUN_TIME, axis=1)[i]) + "\n")

