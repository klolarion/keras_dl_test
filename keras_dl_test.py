import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras import layers, optimizers, losses


def prepare_data():
    iris = load_iris()
    X = iris.data[:, 2:]
    y = iris.target
    lbl_str = iris.target_names
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.20)
    return X_tr, y_tr, X_val, y_val, lbl_str


def visualize(net, X, y, multi_class, labels, class_id, colors, xlabel, ylabel, legend_loc='lower right'):
    # 데이터의 최소~최대 범위를 0.05 간격의 좌표값으로 나열
    x_max = np.ceil(np.max(X[:, 0])).astype(int)
    x_min = np.floor(np.min(X[:, 0])).astype(int)
    y_max = np.ceil(np.max(X[:, 1])).astype(int)
    y_min = np.floor(np.min(X[:, 1])).astype(int)
    x_lin = np.linspace(x_min, x_max, (x_max - x_min) * 20 + 1)
    y_lin = np.linspace(y_min, y_max, (y_max - y_min) * 20 + 1)

    # x_lin과 y_lin의 격자좌표의 x와 y값 구하기
    x_mesh, y_mesh = np.meshgrid(x_lin, y_lin)

    # (x, y) 좌표의 배열로 만들어 신경망의 입력 구성
    X_test = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])

    # 학습된 신경망으로 X_test에 대한 출력 계산
    if multi_class:
        y_hat = net.predict(X_test)
        y_hat = np.array([np.argmax(y_hat[k]) for k in range(len(y_hat))], dtype=int)
    else:
        y_hat = (net.predict(X_test) >= 0.5).astype(int)
        y_hat = y_hat.reshape(len(y_hat))

    # 출력할 그래프의 수평/수직 범위 및 각 클래스에 대한 색상 및 범례 설정
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # 클래스별 산점도 그리기
    for c, i, c_name in zip(colors, labels, class_id):
        # 격자 좌표의 클래스별 산점도
        plt.scatter(X_test[y_hat == i, 0], X_test[y_hat == i, 1],
                    c=c, s=5, alpha=0.3, edgecolors='none')
        # 학습 표본의 클래스별 산점도
        plt.scatter(X[y == i, 0], X[y == i, 1],
                    c=c, s=20, label=c_name)

    # 범례의 표시 위치 지정
    plt.legend(loc=legend_loc)
    # x축과 y축의 레이블을 지정한 후 그래프 출력
    plt.xlabel(xlabel, size=12)
    plt.ylabel(ylabel, size=12)
    plt.show()


nSamples = 150
nDim = 2
nClasses = 3
X_tr, y_tr, X_val, y_val, labels = prepare_data()

# 모델 정의
bp_model_tf = keras.Sequential()
bp_model_tf.add(layers.InputLayer(input_shape=(nDim,)))  # 입력층
bp_model_tf.add(layers.Dense(4, activation='sigmoid'))  # 은닉층
bp_model_tf.add(layers.Dense(nClasses, activation='softmax'))  # 출력층

bp_model_tf.summary()  # 모델의 요약정보 출력

# 모델 훈련을 위한 설정
bp_model_tf.compile(optimizer=optimizers.SGD(0.1, momentum=0.9),
                    loss=losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])

# 모델 학습
bp_model_tf.fit(X_tr, y_tr, batch_size=15, epochs=200, verbose=2, validation_data=(X_val, y_val))

# 훈련된 모델을 이용한 분류
y_hat = bp_model_tf.predict(X_val, verbose=0)
y_hat_lbls = np.array([np.argmax(y_hat[k]) for k in range(len(X_val))])
nCorrect = (y_hat_lbls == y_val).sum()
print('Validation accuracy: {}/{} --> {:7.3f}%'.format(nCorrect, len(X_val), nCorrect * 100.0 / len(X_val)))

#  시각화
visualize(bp_model_tf, X_tr, y_tr,
          multi_class=True,
          class_id=labels,
          labels=[0, 1, 2],
          colors=['red', 'blue', 'green'],
          xlabel='petal length',
          ylabel='petal width')

#  함수형 모델 정의

inputs = keras.Input(shape=(nDim))
h = layers.Dense(4, activation='sigmoid')(inputs)
y = layers.Dense(nClasses, activation='softmax')(h)
bp_model_tf = keras.Model(inputs, y)


# 서브클래싱 모델 정의
class BP_iris(keras.Model):
    def __init__(self):
        super(BP_iris, self).__init__()
        self.h_layer = layers.Dense(4, activation='sigmoid')
        self.o_layer = layers.Dense(nClasses, activation='softmax')

    def call(self, x):
        x = self.h_layer(x)
        return self.o_layer(x)


bp_model_tf = BP_iris()


## 동적학습

#  계단형 감쇠 스케줄러 - epoch구간별 학습률 적용
boundaries = [700, 900]
values = [0.1, 0.05, 0.01]
lr_fn = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
bp_model_tf.compile(optimizer=optimizers.SGD(lr_fn, momentum=0.9),
                    loss=losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])


#  지수함수 감쇠 스케줄러 - 초깃값으로부터 지수함수 형태로 감쇠하는 학습률 적용
lr_fn = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,  # 초기 학습률
    decay_steps=500,  # 변화량적용 epoch횟수
    decay_rate=0.5,  # 변화량   ex) epoch500 -> 0.1 * 0.5 == 0.05 / epoch1000 -> 0.05 * 0.5 = 0.025
    staircase=False)  # True면 정수나눗셈으로 처리


# 다항식 감쇠 스케줄러 - 반복횟수에따라 다항식 함수에 의해 감쇠
lr_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.1,
    decay_steps=900,
    end_learning_rate=0.01,
    power=0.5)

# 임의 스케줄러 클래스
class MyExopnentialDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, decay_steps, decay_rate):
        self.initial_lr = initial_lr
        self.d_s = decay_steps
        self.d_r = decay_rate

    def __call__(self, step):
        return self.initial_lr * self.d_r ** (step / self.d_s)

lr_fn = MyExopnentialDecay(0.1, 900, 0.5)