import numpy as np
import DataSetup

def setup(ver):
    train_x, train_y, test_x, test_y = DataSetup.read_ceps(ver)

    fitted_data, num_labels = train(train_x, train_y)
    accuracy = test(fitted_data, test_x, test_y, num_labels)
    print(accuracy)

    return fitted_data

def train(train_x, train_y):
    sum_x = []
    count_y = []
    fitted_data = []
    for label, y in enumerate(train_y):
        if y == len(sum_x):
            sum_x.append(0)
            count_y.append(0)
        count_y[y] += 1
        #print(train_x, np.mean(train_x))
        sum_x[y] += train_x[label]
    for i in range(len(sum_x)):
        fitted_data.append(sum_x[i]/count_y[i])
    print(len(sum_x), fitted_data)
    return fitted_data, len(sum_x)

def test(fitted_data, test_x, test_y, num_labels):
    accuracy = 0
    for label, y in enumerate(test_y):
        distances = []
        for f_label, f in enumerate(fitted_data):
            distances.append(np.linalg.norm(f - test_x[label]))
        predict_label = np.argmin(distances)
        print("predict_label", predict_label, ":label", y)
        if predict_label == y:
            accuracy += 1

    return accuracy/len(test_x)
