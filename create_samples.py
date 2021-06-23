import numpy as np


class CreateSample:

    def createSample(self, data, shape, features):

        samples_time = []
        targets_time = []

        for j in np.arange(features, len(data), 1):

            sample_temp = []

            for i in range(j - features, j, 1):

                if shape == "vector":
                    sample_temp.append(data[i][0])
                elif shape == "matrix":
                    sample_temp.append(data[i])

            samples_time.append(sample_temp)
            targets_time.append(data[j][0])

        return samples_time, targets_time
