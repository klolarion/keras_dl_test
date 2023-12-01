import tensorflow as tf
import os
from tensorflow import keras
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 학습에 사용될 의류 이미지셋.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("학습 데이터셋의 수: %d" % train_images.shape[0])
print("테스트 데이터셋의 수: %d" % test_images.shape[0])
print("이미지의 크기: %d X %d" % (train_images.shape[1], train_images.shape[2]))
print("정답의 예: %s" % str(train_labels[:20]))
print("학습 이미지의 예: \n%s" % str(train_images[1]))

plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

# 범주 확인
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# 이미지 정규화 - 색상의 범위값으로 나눠 0~1의 실수값으로 정규화한다.
train_images = train_images / 255.0
test_images = test_images / 255.0

# 모델 생성
model = keras.Sequential([
    keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=(3, 3), filters=16),
    # 28x28 크기의 1개 채널 데이터를 받고, 3x3필터를 사용한다. 16개의 특징맵 생성.
    keras.layers.MaxPool2D(strides=(2, 2)),  # 2x2최대풀링 2칸씩 움직이며 연산
    keras.layers.Flatten(),  # 다차원 데이터를 1차원으로 전환
    keras.layers.Dense(128, activation=tf.nn.relu),  # 128개의 뉴런을 가진 완전연결층. relu함수
    keras.layers.Dense(10, activation=tf.nn.softmax)  # 10개의 출력. softmax함수
])

# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습 후 history객체 반환
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))


# 학습정보 함수
def plot_loss(history):
    plt.figure(figsize=(16, 10))
    val = plt.plot(history.epoch, history.history['val_loss'],
                   '--', label='Test')
    plt.plot(history.epoch, history.history['loss'], color=val[0].get_color(),
             label='Train')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.xlim([0, max(history.epoch)])


plot_loss(history)


#성능평가 함수
def evaluation_model(model):
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print('정확도 :', test_accuracy)


evaluation_model(model)

