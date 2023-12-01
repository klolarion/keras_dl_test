import keras.backend
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Layer
import keras.backend as K
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape합수로 데이터 크기 조정 28x28 흑백(채널1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 상하좌우 2픽셀에 0으로 채워진 패딩을 추가.
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

# 경사 소멸문제 방지를 위해 0~1로 데이터 정규화
x_train = x_train / 255
x_test = x_test / 255

# 정수 형태의 레이블 데이터를 to_categorical함수로 벡터화(one-hot encoding)한다.
# 파라미터 10은 레이블의 수를 의미한다. [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] 형태
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# 방사형 기저 함수 RBF 구현.
class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


# 모델 정의( LeNet-5 )
model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=(32, 32, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(RBFLayer(10, 0.5))


# 학습
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test)
print('테스트 정확도 : ', score[1])