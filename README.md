# Image classification-pytorch-cnn
현재 모델 소개:LeNet-5
======================
Yann LeCun 연구팀이 1998년 발표한 CNN알고리즘

개발환경
==================
Jupyter lab, Pytorch

활용 데이터: CIFAR10
====================
내용:10가지 범주를 가진 32x32픽셀의 컬러 이미지 데이터 셋

범주:비행기,자동차,새,고양이,사슴,개,개구리,말,배,트럭

Train set=50000개

Test set=10000개

Chapter
========
1.Import Library

2.Data Download

3.Build Model

4.Model Training

5.Prediction Test data set


1.Import Library
=================
분석에 필요한 라이브러리 Import
![image](https://user-images.githubusercontent.com/104436260/176813424-274dd8fa-9062-4644-8bc7-7d3b2b587308.png)

Seed고정

![image](https://user-images.githubusercontent.com/104436260/176813749-c71c3c24-67f8-4006-b073-87def2ed11d0.png)

2.Data Download
================
transforms.Compose() 활용하여 모듈을 생성하여 후에 이미지 데이터를 불러올 때, 한번에 텐서로 변환 및 정규화 시켜줌
![image](https://user-images.githubusercontent.com/104436260/176813942-29e66e42-84d7-43a5-b737-893ef70bc186.png)


torchivision.dataset에서 데이터 셋 만들기

![image](https://user-images.githubusercontent.com/104436260/176814033-ec1a8396-0327-4c86-b87f-0cafea7f69dc.png)
torchvision.datasets.CIFAR10(root=‘데이터저장위치’, train=True # True=train, False=test,download-True #다운로드 여부,transform=transform  #앞장에 데이터 선처리 작업)

torchvision.dataset.CIFAR10->CIFAR10 데이터 가져오기

torch.utils.data.DataLoader()로 데이터를 불러옴

![image](https://user-images.githubusercontent.com/104436260/176821370-ef23b674-5810-4b53-ab25-fd22a75f4840.png)

batch_size, data shuffle 등 간단히 수행가능

testset도 똑같이 진행

![image](https://user-images.githubusercontent.com/104436260/176821536-be56ba1d-de27-4f8d-b4d8-7ebee3a9c194.png)

CIFAR10 data에 들어있는 class생성

텐서로 변환한 CIFAR10이미지 보여주는 imshow()함수 정의 배치당 4개의 이미지 추출

![image](https://user-images.githubusercontent.com/104436260/176823202-62349f68-e4e0-46c5-8f09-d53180150787.png)

iter()함수로 trainloader에 있는 이미지와 라벨을 순서대로 꺼낼 수 있는 iterator 객체 생성하고, next()함수로 이미지 데이터와 라벨을 
꺼내옴

![image](https://user-images.githubusercontent.com/104436260/176823243-36f515d1-451a-49b2-bbc1-f0b7c0c418e3.png)

images, labels 출력, 이미지는 이미지 정보를 담고있는 4개의 Tensor형태 데이터 추출, labels는 해당 이미지에 대한 라벨값을 0~9사이의 정수로 추출

![image](https://user-images.githubusercontent.com/104436260/176823313-0df01a26-2747-4bb1-b694-d8245f63a2f5.png)

make_grid를 통해 grid tensor를 만들고 이를 이전에 생성한 imshow()함수를 통해 여러 이미지를 출력

3.Build Model
======
Pytorch로 인공신경망 설계
![image](https://user-images.githubusercontent.com/104436260/176824629-f7ee169b-f68b-4368-aba0-7ca5f360fc23.png)

일반적으로 Pytorch로 인공신경망을 설계할 때, nn.Module을 활용하여 만드는 것이 일반적임

1. 상위 클래스를 상속받기 위해, 클래스를 선언할 때 클래스명 뒤에 상속받을 클래스명 입력 ex) Class Net(nn.Module)

2.def__init__(self): 부분에서 함수를 정의함

-super()함수로 상위클래스 상속-> super(모델명,self).__init__()을 통해 nn.Module을 실행시킴

-self.속성명으로 값을 할당함, 여기서는 모델의 구조를 정의함 ex)self.conv1=nn.Conv2d(in_channel, out_channel, filter크기)

self.conv1=convolution layer

self.pool=maxpooling

self.fc1=Fully-connected(3차원에서 1차원으로 변환된 데이터를 계산하여 각 범주에 속할 확률을 계산하는 과정
  
3.forward(): 부분에서 모델이 어떻게 학습할지 코딩

-처음 layer에서 Conv1+ReLu(activation Function)적용 하여 32x32x3 이미지를 28x28x6으로 반환후, Maxpooling 적용(size=2, stride=2)->14x14x6 사이즈로 이미지 반환

-두번째 layer에서 Conv2+ReLu함수를 적용하여 14x14x6 이미지를 10x10x16 사이즈로 반환 후, Maxpooling을 똑같이 적용하여 최종 5x5x16 사이즈로 이미지 반환

-세 번쨰 layer x.view(-1,16*5*5) 부분에서 3차원 형태의 데이터를 1차원 데이터로 flatten작업을 해줌

-네 번째, 다섯 번째 layer에서 flatten된 1차원 데이터에 fully-connected+Relu 함수 적용(fully-connected란 1차원 변환 데이터가 각 범주에 속학확률 계산)

-여섯 번째 layer에서는 fully-connected만 진행, activation function(ReLu)부분 적용하지 않고 return을 통해서 최종 output출력

손실함수&최적화
=======
![image](https://user-images.githubusercontent.com/104436260/176829267-2c306fcb-ec5c-41ad-be92-bdad886ba3b0.png)

criterion은 실제 값과 예측 값의 차이를 수치화한 함수 즉 loss function

CrossEntropyLoss: 다중분류에서 사용하는 손실함수, CIFAR10데이터는 예측할 범주가 10개이기 때문에 다중분류에 해당

Optimizer: loss값을 최소화 해주는 최적의 파라미터를 찿는 알고리즘

SGD: Loss함수의 미분을 이용하여 Loss를 줄이는 것

Learning Rate: 파라미터를 얼마나 세밀하게 조절할 지 정하는 매개변수

Momentum: 파라미터를 업데이트할 때 이전 가중치의 업데이트값의 일정 비율을 반영하는 매개변수

4.Model Training
=================
![image](https://user-images.githubusercontent.com/104436260/176829854-0fa544f1-571f-49e8-8f81-1b5ea979bf9a.png)

-epoch 데이터를 몇 번 반복하여 학습시킬지 결정하는 파라미터

-for문에서 enumerate 인덱스와 원소포함하는 객체로 리턴해줌

-optimizer.zero_grad #한 번 학습이 완료되면, gradient를 0으로 만들어주어야함->초기화하지 않을 시, 의도한 방향과는 다른 방향으로 학습할 가능성이 있음

-순전파+역전파+최적화 부분

-outputs=net(inputs)-> CNN모델 적용 부분

-loss=criterion(outputs, labels) #Loss 계산

-loss.backward() ->backpropogation에서 gradient계산, Loss값들을 가중치들에 대해 미분->가중치1이 변화했을때 함수g가 얼마나 변화했는지 알기 위함

-optimizer.step() ->업데이트할 파라미터와 learning rate을 step() method를 통해 업데이트함.

-학습이 끝난후 트레인 모델 저장 torch.save(net.state_dict(),PATH)->layer마다 텐서로 매핑되는 파라미터들을 딕셔너리 타입으로 저장.

5.Prediction Test data set
==================================
![image](https://user-images.githubusercontent.com/104436260/176832349-237129f9-13a7-4b95-b276-1d848a1e1430.png)
net.load_state_dict(torch.load(PATH)) train으로 학습한 모델의 파라미터를 불러옴

_, predicted = torch.max(outputs, 1) #예측최대값과 최대값에 인데스를 튜플로 반환,행을 기준으로 최대값 찿음->최대값을 기준으로 범주를 예측

![image](https://user-images.githubusercontent.com/104436260/176833052-9c640312-d13c-4854-aeec-d8c57b75f788.png)

torch.no_grad()로 test를 예측할때는 gradient, context 비활성화 시켜줌 필요한 메모리가 줄어들고 연산속도 향상됨

test set전체에 대해 이미지 분류 모델 적용 결과는 60%

![image](https://user-images.githubusercontent.com/104436260/176833109-021bfd0a-7ab8-4ae4-817a-e5a69f5881f9.png)

각 범주별 이미지 분류 정확도 




