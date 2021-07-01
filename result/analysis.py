import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from draw_picture import DrawPicture
from tool import Tool

# 设置文件路径
dir_result = os.getcwd()  # 返回当前文件的路径
dir = os.path.abspath(os.path.dirname(dir_result)) + '\\experiments\\'

# 方法列表
method_list = ["DQN", "DDPG", "DF-DQN"]

# 读取真实能耗值数据
actual_test = pd.read_csv(os.path.abspath(os.path.dirname(dir_result)) + '\\data\\' + "actual_test.csv", header=None)
actual_test = np.array(actual_test)

# 选择一定时间段
time_start = 9 * 24
time_end = 11 * 24

# DF-DQN的起始和终止类别
CLASS = 2
N_CLASS = 16

# 绘画9个子图，返回子图的二维列表
# axes = DrawPicture().drawSubplot(row=3, col=3, width=12, height=8,
#                                  X_label="Time (hour)", Y_label="Energy consumption (kWh)",
#                                  X_labelpad=24, Y_labelpad=36, figName="2016.11.10-2016.11.11", titlepad=24)

axes = DrawPicture().drawSubplot(row=3, col=3, width=12, height=12,
                                 X_label="Predicted energy consumption (kWh)", Y_label="Actual energy consumption (kWh)",
                                 X_labelpad=24, Y_labelpad=36, titlepad=24)

# 9个子图使用的颜色列表
COLOR = ["tomato", "orange", "gold", "limegreen", "darkturquoise", "slategrey", "olive", "violet", "deeppink"]
COLOR_INDEX = 0

# # 绘图的基本设置
# sns.set_theme(context='paper', style="whitegrid")
#
# # 设置坐标轴样式
# font_label = {'family': 'Times New Roman',
#               'weight': 'bold',
#               'size': 12
#               }
#
# # 绘制收敛图的设置
# plt.xlabel("Episode", labelpad=4, fontdict=font_label)
# plt.ylabel("Mean Absolute Error", labelpad=0, fontdict=font_label)

# 绘制一定时刻的真实能耗值
# plt.plot(np.arange(0, len(actual_test[time_start:time_end])), actual_test[time_start:time_end],
#          label="Actual", color="blue")

# 将所有试验误差记录到同一份txt文档中
# filename_result = "result.txt"
# file_log = open(dir + filename_result, 'w')

# 计算DQN，DF-DQN，DDPG预测的平均值
for method in method_list:

    # if method == "DQN":
    #     continue

    # if method == "DDPG" or method == "DQN":
    #     continue

    # if method == "DF-DQN":
    #     continue

    while True:

        if method == "DF-DQN":

            CLASS = CLASS + 1
            dir_method = dir + method + "\\" + "N_CLASS=" + str(CLASS) + "\\" + method + "_mean" + "\\"

        elif method == "DQN" or method == "DDPG":
            dir_method = dir + method + "\\" + method + "_mean" + "\\"

        """
            建立日志，记录运行过程中的情况
        """
        # filename_log = "log.txt"
        # file_log = open(dir_method + filename_log, 'w')

        # if method == "DF-DQN":
        #     file_log.write(method + "，划分 " + str(CLASS) + " 类\n")
        #
        # elif method == "DQN" or method == "DDPG":
        #     file_log.write(method + ":\n")

        """
            读取数据
        """
        # 读取所有次试验的收敛图
        predict_test_mean = pd.read_csv(dir_method + "predict_test_mean.csv", header=None)
        predict_test_mean = np.array(predict_test_mean)

        # mae_train_mean = pd.read_csv(dir_method + "mae_train_mean.csv", header=None)
        # mae_train_mean = np.array(mae_train_mean)

        """
            计算误差
        """
        # 读取所有次试验的预测值，和真实值进行对比
        # 两种，一种是放在每个文件夹下面，一种是集体操作
        # 第一种是放在每个文件夹下面
        # Tool(file_log).mae(actual_test, predict_test_mean)
        # Tool(file_log).rmse(actual_test, predict_test_mean)
        # Tool(file_log).cv(actual_test, predict_test_mean)
        # Tool(file_log).r2(actual_test, predict_test_mean)
        # file_log.write("===================================================\n")

        """
            绘画图像
        """

        # 选择画哪个子图
        subplot_row = int((COLOR_INDEX / 3))
        subplot_col = COLOR_INDEX % 3

        # 九张子图的真实能耗
        # axes[subplot_row, subplot_col].plot(np.arange(0, len(actual_test[time_start:time_end])),
        #                                     actual_test[time_start:time_end], label="Actual", color="blue")

        if method == "DF-DQN":

            figName = "N="
            axes[subplot_row, subplot_col].set_title(figName + str(CLASS))  # 设置图的标题

            # 画单天时刻的图
            # axes[subplot_row, subplot_col].plot(np.arange(0, len(predict_test_mean[time_start:time_end])),
            #                                     predict_test_mean[time_start:time_end], label="N=" + str(CLASS),
            #                                     color=COLOR[COLOR_INDEX])

            # plt.plot(np.arange(0, len(predict_test_mean[time_start:time_end])),
            #          predict_test_mean[time_start:time_end], label="N=" + str(CLASS), color=COLOR[COLOR_INDEX])

            # 绘制第4回合开始后的收敛图
            # plt.plot(np.arange(0, len(mae_train_mean)), mae_train_mean[0:],
            #          label="N=" + str(CLASS), color=COLOR[COLOR_INDEX])

        elif method == "DQN" or method == "DDPG":

            figName = method
            axes[subplot_row, subplot_col].set_title(figName)  # 设置图的标题

            # 画单天时刻的图
            # axes[subplot_row, subplot_col].plot(np.arange(0, len(predict_test_mean[time_start:time_end])),
            #                                     predict_test_mean[time_start:time_end], label=method,
            #                                     color=COLOR[COLOR_INDEX])

            # plt.plot(np.arange(0, len(predict_test_mean[time_start:time_end])),
            #          predict_test_mean[time_start:time_end], label=method, color=COLOR[COLOR_INDEX])

            # plt.plot(np.arange(0, len(mae_train_mean)), mae_train_mean[0:], label=method, color=COLOR[COLOR_INDEX])

        # 三图合一，散点图，误差带，直线 -> 真实能耗和预测能耗图
        x = np.arange(-50, int(np.max(actual_test)))
        axes[subplot_row, subplot_col].scatter(x=predict_test_mean, y=actual_test, c=COLOR[COLOR_INDEX])
        axes[subplot_row, subplot_col].plot(x, 1.2 * x, 'b', linestyle="dashed")
        axes[subplot_row, subplot_col].plot(x, 0.8 * x, 'b', linestyle="dashed")
        axes[subplot_row, subplot_col].fill_between(x=x, y1=1.2 * x, y2=0.8 * x, facecolor=None, alpha=0.1)
        axes[subplot_row, subplot_col].plot(x, x, color="blue")

        # 设定图例
        # leg = plt.legend(loc='upper right')
        # leg.get_frame().set_alpha(1)  # 设置为不透明的背景

        # 设置子图图例
        # leg = axes[subplot_row, subplot_col].legend(loc='upper right')
        # leg.get_frame().set_alpha(0)

        COLOR_INDEX = COLOR_INDEX + 1

        if method == "DF-DQN":
            CLASS = CLASS + 1
            if CLASS == N_CLASS:
                break
        elif method == "DQN" or method == "DDPG":
            break

figName = "trend"
plt.savefig(dir_result + "\\" + figName + '.png')
plt.show()
