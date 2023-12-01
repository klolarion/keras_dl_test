import os

import matplotlib.pyplot as plt
from keras import Sequential, optimizers, callbacks
from keras.layers import Flatten, Dense
from tensorflow import keras
from keras.preprocessing import image
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 학습과정 정확도 및 손실함수 시각화 함수
def plot_metric(h, metric):  # metric 시각화할 척도 'accuracy', 'loss'등 을 지정
    train_history = h.history[metric]  # history객체에서 metric의 척도에 해당되는 값의 리스트를 추출
    val_history = h.history['val_' + metric]
    epochs = range(1, len(train_history) + 1)
    plt.plot(epochs, train_history)
    plt.plot(epochs, val_history)
    plt.legend(['training ' + metric, 'validation ' + metric])
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.show()


# MNIST에서 훈련용 및 평가용 이미지와 레이블 로드
(train_imgs, train_labels), (test_imgs, test_labels) = keras.datasets.mnist.load_data()

# 각 픽셀은 0~255사이의 값을 가지므로 픽셀값을 0~1사이로 정규화.
train_imgs, test_imgs = train_imgs / 255.0, test_imgs / 255.0

# 모델 구성 후 요약 정보 출력
model = Sequential([
    Flatten(input_shape=(28, 28)),  # 28X28이미지를 1차원으로 펼침
    Dense(128, activation='relu'),  # 128개의 뉴런으로 은닉층 구성. 가중치는 기본값인 glorot_normal으로 설정됨.
    Dense(10, activation='softmax')  # 숫자 0~9, 10개의 출력층으로 구성
])
model.summary()

# 조기종료 설정
early_stop = callbacks.EarlyStopping(monitor='val_loss',
                                     min_delta=0,
                                     patience=10,
                                     verbose=1,
                                     restore_best_weights=True)

# 모델 컴파일
model.compile(optimizer=optimizers.RMSprop(0.001, rho=0.9),  # 확률적 경사 하강법
              loss='sparse_categorical_crossentropy',  # 교차 엔트로피
              metrics=['accuracy'])  # 평가를 위한 척도 'accuracy'(정확도)

# 모델 학습
# hist = model.fit(train_imgs, train_labels, epochs=50, validation_split=0.2)
# 조기종료 적용 학습
hist = model.fit(train_imgs, train_labels, epochs=50, callbacks=[early_stop], validation_split=0.2)

# 모델의 손실함수 및 정확도 시각화
plot_metric(hist, 'accuracy')
plot_metric(hist, 'loss')

# 훈련집합 평가
_, train_acc = model.evaluate(train_imgs, train_labels)
print('훈련 테이블 인식률 = ', train_acc)

# 테스트집합 평가
_, test_acc = model.evaluate(test_imgs, test_labels)
print('테스트 테이블 인식률 = ', test_acc)



## 이미지 출력값 오류. 확인 필요

img_path = 'number_img/3.png'
img = keras.preprocessing.image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
img_array = keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # 이미지를 0과 1 사이의 값으로 정규화

predictions = model.predict(img_array)

pre_class = np.argmax(predictions)
print('입력 숫자는? --> ', pre_class)

