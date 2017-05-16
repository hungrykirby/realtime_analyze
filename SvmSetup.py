from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.utils import resample
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

import os
import glob

import DataSetup

def setup(ver):
    train_xyz, train_label, test_xyz, test_label = DataSetup.read_ceps(ver)
    svc = SVC(kernel="linear",
        C=1000.0,
        degree=3,
        gamma='auto',
        cache_size=200,
        class_weight=None,
        coef0=0.0,
        decision_function_shape=None,
        max_iter=-1,
        probability=False,
        random_state=None,
        shrinking=True,
        tol=0.001,
        verbose=False)

    svc.fit(train_xyz, train_label)
    prediction_label = svc.predict(test_xyz)
    cm = confusion_matrix(test_label, prediction_label)

    acc_parcent = accuracy_score(test_label, prediction_label)
    print(acc_parcent)
    print(cm)

    return svc

def stream(svc, xyz):
    predict = svc.predict(xyz)
    #predict = len(x)
    print(predict[0])
