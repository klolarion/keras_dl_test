from tensorflow import keras
from keras import layers

## 가중치의 초기화

# 경사소멸, 경사발산(폭발) 문제를 개선하기위해 가중치를 초기화

# Glorot초기화. fanin, fanout 사용
keras.initializers.GlorotUniform  # 범위의 균등분포로 초깃값 선택. keras기본
dense_layer = layers.Dense(10, activation='relu')

keras.initializers.GlorotNormal  # 평균이 0, 표준편차가 시그마인 정규분포로 초깃값 선택
dense_layer = layers.Dense(10, activation='relu', kernel_initializer='glorot_normal')

# He초기화. fanin 사용
keras.initializers.HeUniform  # 범위의 균등분포로 초깃값 선택.
dense_layer = layers.Dense(10, activation='relu', kernel_initializer='he_uniform')

keras.initializers.HeNormal  # 평균이 0, 표준편차가 시그마인 정규분포로 초깃값 선택
dense_layer = layers.Dense(10, activation='relu', kernel_initializer='he_normal')


## 최적화기 개선

# 네스테로프 가속경사. 모멘텀 사용
keras.optimizers.SGD(0.1, momentum=0.9, nesterov=True)

# Adagrad, RMSProp
# Adagrad는 가속경사법을 개선해서 변화량이 큰 파라미터의 학습률은 낮게, 반대는 크게해서 진행속도를 조정한다.
# RMSPro은 Adagrad의 문제인 학습률이 너무 작아지는 문제를 개선한것으로 학습률 변화를 지수함수적으로 누적한다.
keras.optimizers.RMSprop(0.01, rho=0.9)  # rho - 감쇠를 위한 값

# Adam - 모멘텀과 RMSProp을 결합
# 경사의 이동평균을 베타1로, 경사 제곱의 이동평균을 베타2로 설정하여 사용한다.
# 일반적인 베타값은 0.9 / 0.99를 사용한다
keras.optimizers.Adam(0.01, beta_1=0.9, beta_2=0.99)


## 과적합과 규제

# 조기종료 - validation data set으로 평가하는 경우 손실이 증가하거나 개선되지 않는경우 조기종료한다
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',  # 모니터할 대상.
                                           min_delta=0,  # 개선되는것으로 인정할 최소한의 변화량
                                           patience=0,  # 개선없이 반복할경우 종료할 횟수
                                           mode='auto',  # max, min, auto 모니터할 대상의 증감, 자동 설정
                                           restore_best_weights=True)  # 가장 좋은 지점으 가중치로 되돌릴지 정함

# 가중치 규제 - 복잡도를 낮추기위해 가중치를 작은값으로 조정
# Adamw - 가중치인 w가 커지면 불이익항이 커지고 w값이 작은값으로 유지되게끔 최적화가 진행됨. l2를 주로 사용함
model = layers.Dense(3, activation='relu',
                     kernel_initializer=keras.regularizers.L2(0.01))


# 드롭아웃 - 입력, 은닉층 뉴런 일부를 확률적으로 제외하고 훈련한다. 마지막에는 드롭없이 모든 뉴런이 동작한다.
model = keras.Sequential([keras.Input(shape=(10,)),
                          layers.Dropout(rate=0.2),  # 20%확률로 드롭
                          layers.Dense(32, activation='relu'),
                          layers.Dropout(rate=0.2),
                          layers.Dense(4, activation='softmax')])


## 배치정규화
# SGD에서 각 미니배치마다 평균이 0이고 분산이 1인 정규분포로 변환한다.
model = keras.Sequential([keras.Input(shape=(10,)),
                          layers.BatchNormalization(),  # 배치정규화 진행
                          layers.Dense(32, activation='lelu'),
                          layers.BatchNormalization(),
                          layers.Dense(4, activation='softmax')])