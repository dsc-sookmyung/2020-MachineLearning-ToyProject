# 합성곱 신경망(Convolutional neural network, CNN)
합성곱 신경망은 이미지 처리에 탁월한 성능을 보이는 신경망이다.

합성곱 신경망은 크게 **합성곱층(Convolution layer)과 풀링층(Pooling layer)** 으로 구성된다.

합성곱 연산 후, 합성곱 연산의 결과가 활성화 함수 ReLU를 지난다. 이 두 과정을 합성곱층이라고 한다. 그 후에 POOL이라는 구간을 지나는데 이는 풀링 연산을 의미하며 풀링층이라고 한다.


## 1. 합성곱 신경망의 필요성
다층 퍼셉트론은 몇 가지 픽셀만 값이 달라져도 민감하게 예측에 영향을 받는다는 단점이 있다. 이미지를 다층 퍼셉트론으로 분류한다고 보면, 이미지를 1차원 텐서인 벡터로 변환하고 다층 퍼셉트론의 입력층으로 사용해야 한다. 1차원으로 변환된 결과는 변환 전에 가지고 있던 공간적인 구조(spatial structure) 정보가 유실된 상태이다. 여기서 공간적인 구조 정보라는 것은 거리가 가까운 어떤 픽셀들끼리는 어떤 연관이 있고, 어떤 픽셀들끼리는 값이 비슷하거나 등을 포함하고 있다. 결국 이미지의 공간적인 구조 정보를 보존하면서 학습할 수 있는 방법이 필요해졌고, 이를 위해 사용하는 것이 합성곱 신경망이다.


## 2. 채널(Channel)
이미지 처리의 기본적인 용어.

기계는 글자나 이미지보다 숫자. 다시 말해, 텐서를 더 잘 처리할 수 있다. 이미지는 **(높이, 너비, 채널)** 이라는 3차원 텐서다. 여기서 높이는 세로 방향 픽셀 수, 너비는 이미지의 가로 방향 픽셀 수, 채널은 색 성분을 의미한다.

흑백 이미지는 채널 수가 1, 컬러 이미지는 적색(Red), 녹색(Green), 청색(Blue) 채널 수가 3개다. 채널은 때로는 깊이(depth)라고도 한다. 이 경우 이미지는 **(높이, 너비, 깊이)** 라는 3차원 텐서로 표현된다고 말할 수도 있다.


## 3. 합성곱 연산(Convolution operation)
합성곱층은 합성곱 연산을 통해서 **이미지의 특징을 추출** 하는 역할을 한다. 합성곱은 영어로 convolution 이라고도 불리는데, **커널(kernel)** 또는 **필터(filter)** 라는 n x m 크기의 행렬로 높이 (height) x 너비(width) 크기의 이미지를 처음부터 끝까지 겹치며 훑으면서 n x m 크기의 겹쳐지는 부분의 각 이미지와 커널의 원소의 값을 곱해서 모두 더한 값을 출력으로 하는 것을 말한다. 이때, 이미지의 가장 왼쪽 위부터 가장 오른쪽까지 순차적으로 훑는다.

- 커널(kernel)은 일반적으로 3 x 3 또는 5 x 5를 사용한다.

입력으로부터 커널을 사용하여 합성곱 연산을 통해 나온 결과를 **특성 맵(feature map)** 이라고 한다.

커널의 이동 범위는 **스트라이드(stride)** 라고 한다.


## 4. 패딩(Padding)
합성곱 연산의 결과로 얻은 특성 맵은 입력보다 크기가 작아진다는 특징이 있다. 만약, 합성곱 층을 여러 개 쌓았다면 최종적으로 얻은 특성 맵은 초기 입력보다 매우 작아진 상태가 되어버린다. 합성곱 연산 이후에도 특성 맵의 크기가 입력의 크기와 동일하게 유지되도록 하고 싶다면 패딩(padding)을 사용하면 된다.

패딩은 (합성곱 연산을 하기 전에) 입력의 가장자리에 지정된 개수의 폭만큼 행과 열을 추가해주는 것을 말한다. 좀 더 쉽게 설명하자면 지정된 개수의 폭만큼 테두리를 추가한다. 주로 값을 0으로 채우는 제로 패딩(zero padding)을 사용한다.

만약 스트라이드가 1이라고 하였을 때, 3 x 3 크기의 커널을 사용한다면 1폭짜리 제로 패딩을 사용하고, 5 x 5 크기의 커널을 사용한다면 2폭 짜리 제로 패딩을 사용하면 입력과 특성 맵의 크기를 보존할 수 있다.


## 5. 가중치와 편향
### 1) 합성곱 신경망의 가중치
다층 퍼셉트론과 비교하면서 설명한다.

우선 다층 퍼셉트론으로 3 x 3 이미지를 처리한다고 가정해본다. 이미지를 1차원 텐서인 벡터로 만들면, 3 x 3 = 9가 되므로 입력층은 9개의 뉴런을 갖는다. 그리고 4개의 뉴런을 가지는 은닉층을 추가한다고 해본다. 입력층과 은닉층의 연결선은 가중치를 의미하므로, 9 x 4 = 36개의 가중치를 가진다.

이제 비교를 위해 합성곱 신경망으로 3 x 3 이미지를 처리한다고 해본다. 2 x 2 커널을 사용하고, 스트라이드는 1로 한다. 합성곱 신경망에서 가중치는 커널 행렬의 원소들이다. 최종적으로 특성 맵을 얻기 위해서는 동일한 커널로 이미지 전체를 훑으며 합성곱 연산을 진행한다. 결국 이미지 전체를 훑으면서 사용되는 가중치는 $w_0, w_1, w_2, w_3$ 4개 뿐이다. 그리고 각 합성곱 연산마다 이미지의 모든 픽셀을 사용하는 것이 아니라, 커널과 맵핑되는 픽셀만을 입력으로 사용한다. 결국 합성곱 신경망은 다층 퍼셉트론을 사용할 때보다 훨씬 적은 수의 가중치를 사용하며 공간적 구조 정보를 보존한다는 특징이 있다.

다층 퍼셉트론의 은닉층에서는 가중치 연산 후에 비선형성을 추가하기 위해서 활성화 함수를 통과시킨다. 합성곱 신경망의 은닉층에서도 마찬가지다. 합성곱 연산을 통해 얻은 특성 맵은 다층 퍼셉트론과 마찬가지로 비선형성 추가를 위해서 활성화 함수를 지나게 된다. 이때 렐루 함수나 렐루 함수의 변형들이 주로 사용된다. 이와 같이 합성곱 연산을 통해 특성 맵을 얻고, 활성화 함수를 지나는 연산을 하는 합성곱 신경망의 은닉층을 합성곱 신경망에서는 합성곱 층(convolution layer)이라고 한다.

### 2) 합성곱 신경망의 편향
합성곱 신경망에 편향(bias)를 추가할 수 있다. 만약, 편향을 사용한다면 커널을 적용한 뒤에 더해진다. 편향은 하나의 값만 존재하며, 커널의 적용된 결과의 모든 원소에 더해진다.


## 6. 풀링(Pooling)
일반적으로 합성곱 층(합성곱 연산 + 활성화 함수) 다음에는 풀링 층을 추가하는 것이 일반적이다. 풀링 층에서는 특성 맵을 다운샘플링하여 특성 맵의 크기를 줄이는 풀링 연산이 이뤄진다. 풀링 연산에는 일반적으로 최대 풀링(max pooling)과 평균 풀링(average pooling)이 사용된다.

풀링 연산에서도 합성곱 연산과 마찬가지로 커널과 스트라이드의 개념을 가진다. max pooling은 커널과 겹치는 영역 안에서 최댓값을 추출하는 방식으로 다운샘플링한다.

다른 풀링 기법인 average pooling은 최댓값을 추출하는 것이 아니라 평균값을 추출하는 연산이 된다. 풀링 연산은 커널과 스트라이드 개념이 존재한다는 점에서 합성곱 연산과 유사하지만, 합성곱 연산과의 차이점은 학습해야 할 가중치가 없으며 연산 후에 채널 수가 변하지 않는다는 점이다.

풀링을 사용하면, 특성 맵의 크기가 줄어드므로 특성 맵의 가중치의 개수를 줄여준다.


## Reference
- https://wikidocs.net/64066
- https://stanford.edu/~shervine/l/ko/teaching/cs-230/cheatsheet-convolutional-neural-networks
