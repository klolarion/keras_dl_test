from tensorflow import keras

## cnn층
keras.layers.Conv2D(filters=1,  # 필터 수
                    kernel_size=1,  # 필터의 크기
                    strides=(1, 1),  # 이동간격
                    padding='valid',  # valid - 패딩없음, same - 입력과 출력을 같게 패딩 조정
                    activation=None)  # 활성함수, relu 주로사용


## 풀링층
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None)  # 최대풀링 2차원. 일반적으로 많이 사용.
keras.layers.AveragePooling3D(pool_size=(2, 2), strides=None)  # 평균풀링 3차원


## 완전연결층
keras.layers.Flatten(data_format=None)  # flatten층. 입력데이터를 1차원 데이터로 변환. ex) 3x3 -> 1x9
keras.layers.Dense(units=None, activation=None)  # units - 뉴런 갯수
keras.layers.Dense(units=None, activation='softmax')  # 출력층

