import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class DrawPicture:
    """
        根据传入的数据绘制不同的图像，loss、reward图等
    """

    # def __init__(self):
    # mpl.rcParams['font.sans-serif'] = ['SimHei']  # 作图显示中文

    def Xpredicted_Yactual(self, dir, figName, action_predict, action_true):
        """
            1. 绘制预测动作值和真实动作值结果图，横坐标是预测值，纵坐标真实值

            Args:
                figName: 图像标题
                action_predict: 预测的动作值
                action_true: 真实的动作值
        """
        plt.title(figName)

        # marker:表示的是标记的样式，默认的是'o'
        # x轴用预测的动作值，y轴用真实的动作值，绘制一个散点图
        plt.scatter(action_predict, action_true, marker='.')
        plt.xlabel('Predicted Value')  # 设置x轴标签
        plt.ylabel('Actual value')  # 设置y轴标签

        plt.savefig(dir + "\\" + figName + '.png')
        plt.show()

    def Xrange_Ypredicted_Yactual(self, dir, figName, action_predict, action_true):
        """
            2. 绘制真实值和预测值两个图像
                1. x轴是range，y轴是真实值
                2. x轴是range，y轴是预测值

            Args:
                figName: 图像标题
                action_predict: 预测的动作值
                action_true: 真实的动作值
        """

        plt.title(figName)  # 设置图的标题

        x = np.arange(0, len(action_true))  # 设置x轴

        plt.plot(x, action_predict, label='predict value')
        plt.plot(x, action_true, label='true value')

        leg = plt.legend(loc='upper left')  # 图例在左上角，
        leg.get_frame().set_alpha(1)  # 设置为不透明的背景

        plt.savefig(dir + "\\" + figName + '.png')
        plt.show()

    def Xrange_Y(self, dir, figName, Yname, Y):
        """
            3. 绘制每个步骤得到的奖赏值

            Args:
                figName: 图像标题
                reward: 奖赏值数组
        """
        plt.title(figName)

        x = np.arange(0, len(Y))

        # if Yname == "reward":
        #     fig.gca().invert_yaxis()  # 翻转y轴,越往上越负

        plt.plot(x, Y, label=Yname)

        plt.savefig(dir + "\\" + figName + '.png')
        plt.show()

    def Xthousand_Ypredicted_Yactual(self, dir, figName, action_predict, action_true):
        """
            4. 前一千步的真实值和预测值对比图，和最后一千步的真实值和预测值对比图

            Args:
                figName: 图像标题
                action_predict: 预测的动作值
                action_true: 真实的动作值
        """

        ax = plt.subplot(2, 1, 1)  # 画在图像上半部分的前一千步的真实值和预测值对比图
        plt.title(figName)  # 设置图的标题

        x = np.arange(0, 20, 1)  # x轴 1000步

        plt.plot(x, action_predict[0:20], label='predict value')  # 前一千步的预测值
        plt.plot(x, action_true[0:20], label='true value')  # 前一千步的真实值

        leg = plt.legend(loc='upper left')  # 图例在左上角
        leg.get_frame().set_alpha(1)  # 设置为不透明的背景

        ax = plt.subplot(2, 1, 2)  # 画在图像下半部分的最后一千步的真实值和预测值对比图

        plt.plot(x, action_predict[-20:], label='predict value')  # 最后一千步的预测值
        plt.plot(x, action_true[-20:], label='true value')  # 最后一千步的真实值

        leg = plt.legend(loc='upper left')
        leg.get_frame().set_alpha(1)

        plt.savefig(dir + "\\" + figName + '.png')
        plt.show()

    def XrangeError(self, dir, figName, action_predict, action_true):
        """
            画出误差图
        """
        plt.title(figName)
        error = []
        for i in range(len(action_true)):
            error.append(abs(action_true[i] - action_predict[i]))

        x = np.arange(0, len(action_true))
        plt.plot(x, error)

        plt.savefig(dir + "\\" + figName + '.png')
        plt.show()

    def drawSubplot(self, row, col, width, height, X_label, Y_label, X_labelpad, Y_labelpad, figName=None,
                    titlepad=None):
        """
            Args:
                row: 子图的行数
                col: 子图的列数
                width: 主图的宽度
                height: 主图的高度
                X_label: 主图的横轴标签
                Y_label: 主图的纵轴标签
                X_labelpad: 主图的横轴标签的间距
                Y_labelpad: 主图的纵轴标签的间距
                figName: 主图的图像名称
                titlepad: 主图的图像标题间距

            Returns:
              axes: 子图列表
        """

        # 绘图的基础风格设置
        sns.set_context(context='paper')

        # 将所有的图像绘制在一起, 3*3的图像
        fig = plt.figure(figsize=(width, height))  # dpi=None，调整像素

        # 设置坐标轴样式
        font_label = {'family': 'Times New Roman',
                      'weight': 'bold',
                      'size': 16
                      }

        plt.title(figName, fontdict=font_label, pad=titlepad)
        plt.xlabel(X_label, labelpad=X_labelpad, fontdict=font_label)
        plt.ylabel(Y_label, labelpad=Y_labelpad, fontdict=font_label)

        # 删除第一张图的x轴y轴刻度值
        plt.xticks([])
        plt.yticks([])

        # 去除外边框的线，这里有两张图，第一张是整体的图，第二张是九个子图，所以在设置时会混乱，此时去除的边框线是第一张图的所有边框线
        sns.despine(top=True, bottom=True, left=True, right=True)

        # 对每个子图使用 whitegrid 的绘图风格
        with sns.axes_style("whitegrid"):
            axes = fig.subplots(row, col)  # 3*3的图像

        return axes
