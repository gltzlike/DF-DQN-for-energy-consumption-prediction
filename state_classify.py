import numpy as np
from sklearn.metrics import accuracy_score
from deepforest import CascadeForestClassifier


class StateClassify:

    def constructState(self, data_train_scale, data_test_scale, class_train_true, class_test_true, file_log):

        model = CascadeForestClassifier()
        model.fit(data_train_scale, class_train_true)

        class_train_pre = model.predict(data_train_scale)
        class_test_pre = model.predict(data_test_scale)

        class_train_proba = model.predict_proba(data_train_scale)
        class_test_proba = model.predict_proba(data_test_scale)

        acc_test = accuracy_score(class_test_true, class_test_pre) * 100

        print("acc_test: {:.3f} %".format(acc_test))
        file_log.write("acc_test: {:.3f} %\n\n".format(acc_test))

        state_train_scale = np.hstack((data_train_scale, class_train_proba))
        state_test_scale = np.hstack((data_test_scale, class_test_proba))

        return state_train_scale, state_test_scale, class_train_pre, class_test_pre
