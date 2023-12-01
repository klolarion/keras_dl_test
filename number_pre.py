# 딥러닝에 필요한 케라스 함수 호출
import numpy as np

# 필요 라이브러리 호출(PIL은 이미지파일 처리위함)
from PIL import Image
from keras.models import load_model

# test.png는 그림판에서 붓으로 숫자 8을 그린 이미지 파일
# test.png 파일 열어서 L(256단계 흑백이미지)로 변환
img = Image.open("number_img/4.png").convert("L")

# 이미지를 784개 흑백 픽셀로 사이즈 변환
img = np.resize(img, (1, 784))

# 데이터를 모델에 적용할 수 있도록 가공
test_data = ((np.array(img) / 255) - 1) * -1

# 모델 불러오기
model = load_model('Predict_Model.h5')

# 클래스 예측 함수에 가공된 테스트 데이터 넣어 결과 도출
# res = model.predict_classes(test_data)
# 2021/10/02 수정 - 오류시 아래 명령어로 대체 가능합니다.
res = (model.predict(test_data) > 0.5).astype("int32")

# 길이10 리스트. 1의 인덱스가 출력된 숫자.
print(res)
