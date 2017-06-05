import numpy as np
import matplotlib.pyplot as plt
from chainer import Variable,FunctionSet,optimizers
import chainer.functions as F
import chainer.links as L
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

import SvmSetup
import DataSetup

class DeepLearning:
    def __init__(self, batchsize, n_epoch):
        # バッチサイズ
        self.batchsize = batchsize
        # エポック数
        self.n_epoch = n_epoch
        # 最適化手法
        self.optimizer = optimizers.Adam()

    def forward(self, x_data, y_data=None, train=True):
        x = Variable(x_data)
        b1 = self.model.bn1(x)
        h1 = F.dropout(F.relu(self.model.l1(b1)), ratio=0.2, train=train)
        b2 = self.model.bn2(h1)
        h2 = F.dropout(F.relu(self.model.l2(b2)), ratio=0.2, train=train)
        b3 = self.model.bn3(h2)
        y = F.dropout(F.relu(self.model.l3(b3)), ratio=0.2, train=train)

        if train:
            t = Variable(y_data)
            return F.softmax_cross_entropy(y,t), F.accuracy(y,t), y.data

        else:
            return y.data

    def fit(self, x_train, y_train):

        self.number_of_label=np.unique(y_train).size
        self.model = FunctionSet(
            bn1 = L.BatchNormalization(x_train.shape[1]),
            l1 = F.Linear(x_train.shape[1], 1000),
            bn2 = L.BatchNormalization(1000),
            l2 = F.Linear(1000, 1000),
            bn3 = L.BatchNormalization(1000),
            l3 = F.Linear(1000, self.number_of_label)
            )

        # トレーニングデータの数
        N = y_train.size
        # optimizer セットアップ
        self.optimizer.setup(self.model.collect_parameters())

        # 念のためトレーニングデータの型合わせ
        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.int32)

        # Learning loop
        for epoch in range(1,self.n_epoch+1):
            # エポック数表示
            print('epoch{}'.format(epoch))

            # training
            # N個の順番をランダムに並び替える
            perm = np.random.permutation(N)

            # ロスと正答率初期化
            sum_accuracy = 0
            sum_loss = 0
            # 0〜Nまでのデータをバッチサイズごとに使って学習
            for i in range(0, N, self.batchsize):
                if i + self.batchsize < N:
                    x_batch = x_train[perm[i:i+self.batchsize]]
                    y_batch = y_train[perm[i:i+self.batchsize]]

                else:
                    x_batch = x_train[perm[i:]]
                    y_batch = y_train[perm[i:]]

                # 勾配を初期化
                self.optimizer.zero_grads()
                # 順伝播させて誤差と精度を算出
                loss, acc, answer = self.forward(x_batch, y_batch)
                # 誤差逆伝播で勾配を計算
                loss.backward()
                self.optimizer.update()

                sum_loss     += float(loss.data) * x_batch.shape[0]
                sum_accuracy += float(acc.data) * x_batch.shape[0]

            print('train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N))

    def predict(self, x_test):
        N_test=x_test.shape[0]
        # ロスと正答率初期化
        sum_accuracy = 0
        sum_loss = 0
        answers = np.array([])
        x_test = x_test.astype(np.float32)
        for i in range(0, N_test, self.batchsize):
            if i + self.batchsize < N_test:
                x_batch = x_test[i:i+self.batchsize]

            else:
                x_batch = x_test[i:]

            # 順伝播させて出力
            sub_answer = self.forward(x_batch, train=False)
            # 出力最大となるラベルを予測値とする
            sub_answer = np.argmax(sub_answer, axis=1)
            answers = np.append(answers, sub_answer)

        answers=answers.reshape(-1,1)

        return answers

    def predict_proba(self, x_test):
        N_test=x_test.shape[0]
        # ロスと正答率初期化
        sum_accuracy = 0
        sum_loss = 0
        answers = np.zeros((x_test.shape[0], self.number_of_label))
        x_test = x_test.astype(np.float32)
        for i in range(0, N_test, self.batchsize):
            if i + self.batchsize < N_test:
                x_batch = x_test[i:i+self.batchsize]
                answers[i:i+self.batchsize] = self.forward(x_batch, train=False)

            else:
                x_batch = x_test[i:]
                answers[i:] = self.forward(x_batch, train=False)

        probability=np.exp(answers)/np.sum(np.exp(answers), axis=1).reshape(-1,1)
        return probability

    def score(self, x_test, y_test):
        predict_y = self.predict(x_test).reshape(-1,)
        wrong_size = np.count_nonzero(predict_y - y_test)
        score=float(y_test.size - wrong_size)/y_test.size
        return score

def setup(file_type, ver):
    x_train, y_train, x_test, y_test = DataSetup.read_ceps(file_type, ver)
    #print(DataSetup.read_ceps(ver))
    # 分類器定義
    DL = DeepLearning(2000, 100)
    # トレーニング
    DL.fit(x_train, y_train)

    # 予測
    predict_answer = DL.predict(x_test)
    print(predict_answer)
    # 各ラベルをとる確率の計算
    probability = DL.predict_proba(x_test)
    print(probability)
    # テストデータでの正答率
    score = DL.score(x_test, y_test)
    print(score)

    return DL

def stream(DL, xyz):
    predict = DL.predict(xyz)
    print(predict[0])
