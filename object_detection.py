'''
아래의 Torchy 클래스는
img 폴더 내에서 사진 10장에 대한 object detection(객체 감지)를 수행합니다.
'''

# 필요한 라이브러리 import하기
# pip install torch
# pip install torchvision

import torch
import torchvision
import sys 
import os
import cv2 

from torchvision import transforms as T
from PIL import Image 

# img 폴더 밑에 있는 1번 이미지 경로 조사
script_dir = os.path.realpath(__file__)
img_file_dir = os.path.normpath(os.path.join(script_dir, "../img/3_dog_and_cat.jpg"))

# DNN의 가장 유명한 모델 中 하나인 Faster R-CNN 모델 채택 후 이미 트레인된 설정값을 넣어주고, 
# 각 픽셀의 근삿값 예측시작을 위한 준비 완료
# 구체적인 설명은 하기 링크를 참조:https://www.notion.so/serene2/Deep-Learning-DNN-c3609aa05ab2431785e1021501c2851f
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# 모델의 아키텍처를 확인
model.eval()

# Image 라이브러리를 이용한 이미지 객체 열기(생성)
ig = Image.open(img_file_dir)

# pytorch 모델에 삽일 할 이미지 객체를 Tensor 포맷으로 변환
transform = T.ToTensor()
img = transform(ig)

# pytorch의 gradient 함수를 사용하지 않고 img 객체에 대한 예측값을 계산
with torch.no_grad():
    pred = model([img])

# [선택] pred을 출력하면 dict을 안고 있는 list이며, 3가지 component가 존재: bounding 박스와 label과 score등이 그것
print(pred)
print(pred[0])
print(pred[0].keys())

# 각각 필요한 요소에 할당하기
# bboxes를 출력해보면, 각각 4가지 수치가 나오는데, 그것은 object bounding box의 좌측 상단의 (x1, y1) 좌표 + 우측 하단의 (x2, y2) 좌표를 나타냄
bboxes, labels, scores = pred[0]['boxes'], pred[0]['labels'], pred[0]['scores']

# score들은 순서가 큰 것 부터 내림차순이며, 해당 조건을 만족하는 entity의 개수만 조사
num = torch.argwhere(scores > 0.7).shape[0]

# object detect을 할 때 기준이 되는 label들을 미리 분류
label_names = ["cat" , "dog" , "man" , "woman" , "airplane" , "bus" , "train" , "truck" , "boat" , "traffic light" , "fire hydrant" , "street sign" , "stop sign" , "parking meter" , "bench" , "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , "elephant" , "bear" , "zebra" , "giraffe" , "hat" , "backpack" , "umbrella" , "shoe" , "eye glasses" , "handbag" , "tie" , "suitcase" , 
"frisbee" , "skis" , "snowboard" , "sports ball" , "kite" , "baseball bat" , 
"baseball glove" , "skateboard" , "surfboard" , "tennis racket" , "bottle" , 
"plate" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" , 
"banana" , "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog" ,
"pizza" , "donut" , "cake" , "chair" , "couch" , "potted plant" , "bed" ,
"mirror" , "dining table" , "window" , "desk" , "toilet" , "door" , "tv" ,
"laptop" , "mouse" , "remote" , "keyboard" , "cell phone" , "microwave" ,
"oven" , "toaster" , "sink" , "refrigerator" , "blender" , "book" ,
"clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush" , "hair brush"]

# computer vision 라이브러리 中 하나인 cv2을 이용해서 이미지 객체를 읽어오고 bbox의 폰트설정
igg = cv2.imread(img_file_dir)
font = cv2.FONT_HERSHEY_COMPLEX

# score 기준에 부합한 object에 한해서 반복문 수행하면서, bbox의 4가지 좌표(x1,y1,x2,y2)를 할당 --> tensor 포맷에서 numpy 포맷으로 변환
for i in range(num):
    x1, y1, x2, y2 = bboxes[i].numpy().astype("int")
    # print(x1, y1, x2, y2)
    # break
    
    # labels을 선회하면서, 순서에 맞는 class_name을 할당
    class_name = label_names[labels.numpy()[i]-1]
    
    # 사각형 bounding box의 좌표를 할당해서 두께가 1인 초록색 테두리 박스를 생성
    igg = cv2.rectangle(igg, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # 매칭되는 class name을 좌측 상단에 파랑색으로, FONT_HERSHEY_COMPLEX font을 사용해서 명시한다.
    igg = cv2.putText(igg, class_name , (x1, y1-10), font, 0.5, (255, 0, 0), 1 , cv2.LINE_AA)

# 완성된 cv2 객체를 보여준다
cv2.imshow('3_dog_and_cat',igg)
cv2.waitKey()