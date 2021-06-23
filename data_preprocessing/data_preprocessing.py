import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chinese_calendar import is_workday
from sklearn.neighbors import LocalOutlierFactor


class DataPreprocessing:

    def __init__(self):

        self.dir = os.getcwd()
        self.data_dir = self.dir + "/data/"

    def segQW(self, load_filename, build_filename, building_id, energy_type):

        """
            The source data (6 train.csv) includes 0-19 buildings, and each building has two energy consumption data types, Q and W.
            A certain data type of one of the buildings is selected for segmentation

            Q: light socket
            W: air conditioner

            Args:
                load_filename: filename of source data
                build_filename: filename of generated data
                building_id: 0 - 19
                energy_type: Q or W

        """

        """
            Data sample:
                'Time', '2016-02-20 10:00:00'
                'BuildingID', '1'
                'Type', 'W'
                'Record', '36.98'
        """

        time = []
        record = []

        # load data
        with open(self.data_dir + load_filename) as data_file:
            data = csv.DictReader(data_file)
            for row in data:
                if row['BuildingID'] == building_id:
                    if row['Type'] == energy_type:
                        time.append(row['Time'])
                        record.append(row['Record'])

        # save data
        data = pd.DataFrame([time, record])
        data = data.transpose()

        data.to_csv(self.data_dir + build_filename, index=False, header=None)

    def outliersProcess(self, load_filename, build_filename, n_neighbors):

        """
            Outliers are detected and replaced

            Args:
                load_filename: filename of source data
                build_filename: filename of generated data
                n_neighbors: hyper-parameters of LOF algorithm

        """

        # load the source data with outliers
        raw_data = pd.read_csv(self.data_dir + load_filename, header=None, nrows=16056)  # 前16056行数据，2015/1/1-2016/10/31
        date = pd.to_datetime(raw_data[0])
        data = np.array(raw_data[1])  # 读取data中的能耗属性列
        data = data.reshape((-1, 1))  # 将能耗数据转换为向量形式

        plt.plot(date, data)
        plt.show()

        # create log
        log_name = "log_outliers.txt"
        if os.path.exists(log_name):  # 文件存在则删除，不存在就写入
            os.remove(log_name)
        log_file = open(log_name, 'w')

        # detect negative data
        index_negative = []  # 负值的索引列表
        for index in range(len(data)):
            if data[index] <= 0:
                index_negative.append(index)

        # process negative data
        for index in index_negative:
            data[index] = self.outliersReplacement(date, data, index, index_negative, log_file)

        # detect positive data by lof
        lof_clf = LocalOutlierFactor(n_neighbors=n_neighbors)  # 创建lof实例
        pred = lof_clf.fit_predict(data)                       # 使用lof对数据进行密度检测
        s = lof_clf.negative_outlier_factor_                   # lof的negative_outlier_factor_属性，算出的值越靠近-1，说明越好

        plt.title("lof")
        plt.scatter(date, data, c=pred)
        plt.show()

        outliers_lof = []   # outliers List

        for index in range(len(pred)):
            if pred[index] == -1:
                outliers_lof.append(index)

        # process positive data
        for index in outliers_lof:
            data[index] = self.outliersReplacement(date, data, index, outliers_lof, log_file)

        log_file.close()

        plt.plot(date, data)
        plt.show()

        # save data
        data = pd.DataFrame(data)
        if os.path.exists(build_filename):  # 文件存在则删除，不存在就写入
            os.remove(build_filename)
        data = pd.concat([date, data], axis=1)
        data.to_csv(self.data_dir + build_filename, index=False, header=None)

    def outliersReplacement(self, date, data, index, outliers, log_file):

        """
            Outliers are replaced

        """

        log_file.write("before replaced: " + str(date[index]) + str(data[index]) + '\n')

        data_replace = 0

        left = 0
        right = 0

        for k in np.arange(1, 365):

            if (index - 24 * k) >= 0:

                if ((is_workday(date[index]) == is_workday(date[index - 24 * k])) and (
                        data[index - 24 * k] >= 0) and not (
                        (index - 24 * k) in outliers)):
                    left = data[index - 24 * k]
                    log_file.write("left normal energy: " + str(date[index - 24 * k]) + str(data[index - 24 * k]) + '\n')
                    break

                else:
                    continue
            else:
                break

        for k in np.arange(1, 365):
            if (index + 24 * k) < len(data):
                if (is_workday(date[index]) == is_workday(date[index + 24 * k]) and (
                        data[index + 24 * k] >= 0) and not (
                        (index + 24 * k) in outliers)):
                    right = data[index + 24 * k]
                    log_file.write("right normal energy: " + str(date[index + 24 * k]) + str(data[index + 24 * k]) + '\n')
                    break
                else:
                    continue
            else:
                break

        if left == 0:
            data_replace = right
        if right == 0:
            data_replace = left
        if left != 0 and right != 0:
            data_replace = (left + right) / 2

        log_file.write("after replaced: " + str(data_replace) + '\n')
        log_file.write("averge: " + str((left + right) / 2) + '\n' + '\n')

        return data_replace

    def joint(self, left_filename, right_filename, build_filename):

        """
            The processed data (2015/1/1-2016/10/31) are stitched together with the unprocessed data (2016/11/1-2016/12/31)

            Args:
                left_filename: processed data (2015/1/1-2016/10/31)
                right_filename: unprocessed data (2016/11/1-2016/12/31)
                build_filename: filename of joint data
        """

        # processed data (2015/1/1-2016/10/31)
        data_left = pd.read_csv(self.data_dir + left_filename, header=None)

        # unprocessed data (2016/11/1-2016/12/31)
        data_right = pd.read_csv(self.data_dir + right_filename, header=None)
        data_right = data_right[16056:17520]

        data_right[0] = pd.to_datetime(data_right[0])
        data = pd.concat([data_left, data_right], axis=0)  # 行数增加的拼接

        if os.path.exists(build_filename):
            os.remove(build_filename)

        data.to_csv(self.data_dir + build_filename, index=False, header=None)