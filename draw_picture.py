import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class DrawPicture:

    def Xrange_Y(self, dir, figName, Yname, Y):

        plt.title(figName)

        x = np.arange(0, len(Y))

        # if Yname == "reward":
        #     fig.gca().invert_yaxis()

        plt.plot(x, Y, label=Yname)

        plt.savefig(dir + "\\" + figName + '.png')
        plt.show()

    def Xrange_Ypredicted_Yactual(self, dir, figName, action_predict, action_true):

        plt.title(figName)
        x = np.arange(0, len(action_true))

        plt.plot(x, action_predict, label='predict value')
        plt.plot(x, action_true, label='true value')

        leg = plt.legend(loc='upper left')
        leg.get_frame().set_alpha(1)

        plt.savefig(dir + "\\" + figName + '.png')
        plt.show()