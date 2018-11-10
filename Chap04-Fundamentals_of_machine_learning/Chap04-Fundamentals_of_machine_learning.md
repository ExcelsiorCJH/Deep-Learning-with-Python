# Chap04 - 머신러닝의 기본 요소



## Goals

- 모델 평가, 데이터 전처리, 특성 공학, 오버피팅에 대해 알아본다.
- 머신러닝 문제 해결을 위한 작업 프로세스에 대해 알아본다.



## 4.1 머신러닝의 네 가지 분류

- 일반적으로 머신러닝 알고리즘은 아래의 그림과 같이 4개의 범주안에 속한다.

![](./images/types.png)



### 4.1.1 지도 학습 (Supervised Learning)

- 샘플 데이터가 주어지면 알고있는 타겟(레이블)에 입력 데이터를 매핑하는 방법을 학습한다.
- 문자 판독, 음성 인식, 이미지 분류, 언어 번역 등이 지도학습에 속한다.
- 지도학습은 대부분 분류(classification)와 회귀(regression)로 구성되지만 다음과 같은 특이한 변종도 많다.
  - **시퀀스 생성**(sequence generation) : 사진이 주어지면 이를 설명하는 캡션을 생성한다. 시퀀스 생성은 일련의 분류 문제로 재구성할 수 있다.
  - **구문 트리**(syntax tree) **예측** : 문장이 주어지면 분해된 구문 트리를 예측한다.
  - **물체 감지**(object detection) : 사진이 주어지면 사진 안의 특정 물체 주위에 경계 상자(bounding box)를 그린다. 이것은 분류 문제로 표현되거나, 경계 상자의 좌표를 벡터 회귀로 예측하는 회귀 + 분류가 결합된 문제로 표현될 수 있다.
  - **이미지 분할**(image segmentation) : 사진이 주어졌을 때 픽셀 단위로 특정 물체에 마스킹(masking)을 한다.



### 4.1.2 비지도 학습 (Unsupervised Learning)

- 타겟(레이블)을 사용하지 않고, 입력 데이터에 대해 유의미한 정보를 찾는 방법이다.
- 데이터 시각화, 데이터 압축, 데이터의 노이즈 제거, 데이터에 있는 상관관계를 더 잘 이해하기 위해 사용한다.
- 비지도 학습으로는 대표적으로 **차원 축소**(dimensionality reduction)와 **군집**(clustering)이 있다.



### 4.1.3 준지도 학습 (Semi-supervised Learning)

- 전체 데이터 중에서 일부에만 레이블되어 있는 경우에 대해 학습하는 방법을 말한다.
- 대부분의 준지도 학습은 지도 학습과 비지도 학습의 조합으로 이루어진다.



### 4.1.4 자기 지도 학습 (Self-supervised Learning)

- 자기 지도 학습은 지도 학습의 특별한 경우라고 할 수 있다.
- 자기 지도 학습은 지도 학습이지만 사람이 만든 레이블을 사용하지 않는다.
- 레이블이 필요하지만 보통 경험적인 알고리즘(heuristic algorithm)을 사용해서 입력 데이터로부터 생성한다.
  - **오토인코더**(AutoEncoder)에서의 레이블은 입력 데이터 자신이다.
  - 지난 프레임이 주어졌을 때 비디오의 다음 프레임을 예측
  - 이전 단어가 주어졌을 때 다음 단어를 예측
- 지도 학습, 자기 지도 학습, 비지도 학습의 구분은 때때로 모호할 수 있다.
  - 오토인코더의 경우 타겟이 있으므로, (자기)지도학습으로 볼 수 있지만, 입력 데이터의 차원 축소 용도로 사용될 때는 비지도 학습으로 볼 수 있다.



### 4.1.5 강화 학습 (Reinforcement Learning)

- 강화 학습에서 **에이전트**(agent)는 환경에 대한 정보를 받아 보상을 최대화하는 행동을 선택하도록 학습한다.

<img src="./images/rl.png" height="50%" width="50%"/>





### 4.1.6 분류와 회귀에서 사용하는 용어

| 용어                                                         | 설명                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **샘플(sample)** 또는 **입력(input)**                        | 모델에 주입될 하나의 데이터 포인트                           |
| **예측(predict)** 또는 **출력(output)**                      | 모델로부터 나오는 값                                         |
| **타겟(Target)**                                             | 정답, 모델이 완벽하게 예측해야 하는 값                       |
| **예측 오차(prediction error)** 또는 **손실 값(loss value)** : | 모델의 예측과 타깃 사이의 거리를 측정한 값                   |
| **클래스(class)**                                            | 분류 문제에서 선택할 수 있는 레이블의 집합                   |
| **레이블(label)**                                            | 분류 문제에서 클래스 할당의 구체적인 사례                    |
| **Ground-truth** 또는 **Annotation**                         | 데이터에셋에 대한 모든 타겟. 일반적으로 사람에 의해 수집된다. |
| **이진 분류**(Binary classification)                         | 각 입력 샘플이 2개의 범주로 구분되는 분류 작업               |
| **다중 분류**(Multiclass classification)                     | 각 입력 샘플이 2개 이상의 범주로 구분되는 분류 작업          |
| **다중 레이블 분류**(Multilabel classification)              | 각 입력 샘플이 여러 개의 레이블에 할당될 수 있는 분류 작업. 예를 들어 하나의 이미지에 고양이와 개가 있을 경우 '고양이'레이블과 '강아지'레이블이 모두 할당되어야 한다. |
| **스칼라 회귀**(Scalar regression)                           | 타겟이 연속적인 스칼라 값인 작업                             |
| **벡터 회귀**(Vector regression)                             | 타겟이 연속적인 값의 집합인 작업. 여러 값에 대한 회귀        |
| **미니 배치** 또는 **배치**(Mini-batch)                      | 모델에 의해 동시에 처리되는 소량의 샘플 묶음. 일반적으로 8 ~ 128이며, 샘플의 개수는 GPU의 메모리 할당이 용이하도록 2의 거듭제곱으로 한다. 훈련할 때 미니 배치마다 경사 하강법 업데이트 값을 계산한다. |





## 4.2 머신러닝 모델 평가

- 머신러닝의 목표는 처음 존 데이터에서 잘 작동하는 **일반화(generalized)**된 모델을 얻는 것이다.



### 4.2.1 훈련, 검증, 테스트 셋

- 모델 평가의 핵심은 데이터를 항상 훈련(training), 검증(valid), 테스트(test) 3개의 세트로 나누는 것이다.
  - 훈련 세트(training set)에서 모델을 훈련하고,
  - 검증 세트(validation set)에서 모델을 평가한다.
  - 테스트 세트를 이용해 모델을 테스트 한다.
- 전체 데이터 셋을 훈련과 테스트 2개만으로 나누지 않는 이유는, 모델을 개발할 때 항상 모델의 설정을 튜닝하기 때문이다.
  - 예를 들어, 층의 수나 층의 유닛 수를 조정할 수 있다. → 이러한 파라미터를 **하이퍼파라미터**(hyperparameter)라고 한다.
- 검증 세트에서 모델의 성능을 평가하여 이런 튜닝을 수행한다.
  - 이러한 튜닝 또한 어떠한 파라미터 공간에서 좋은 설정값을 찾는 **학습**이라 할 수 있다.
- 검증 세트를 이용해 설정을 튜닝하게 되면, **검증 세트에 오버피팅**될 수 있다.
  - 이러한 현상은 검증 세트의 모델 성능에 기반하여 모델의 하이퍼파라미터를 조정할 때마다 검증 데이터에 관한 정보가 모델로 새기 때문이다. → **정보 누설**(information leak)
- 따라서, 모델이 처음 본 데이터인 테스트 세트를 이용하여 모델을 평가한다.
  - 모델은 간접적으로라도 테스트 세트에 대한 어떠한 정보도 얻어서는 안된다.
- 데이터를 훈련/검증/테스트 세트로 나누는 대표적인 방법으로는 다음과 같이 세가지 방법이 있다.
  - 홀드아웃 검증 (hold-out validation)
  - K-폴드 교차 검증 (K-fold cross-validation)
  - 셔플링(shuffling)을 사용한 iterated K-fold cross-validation



#### Simple Hold-Out Validation

- 전체 데이터 셋에서 일정량을 테스트 셋으로 떼어 놓는다.
- 남은 데이터에서 훈련하고 검증 세트로 평가한다.



![hold-out-validation](./images/hold-out.PNG)



- `NumPy`를 이용한 홀드아웃 검증 구현 코드는 다음과 같다.

```python
import numpy as np

num_validation_samples = 10000

np.random.shuffle(data)  # 데이터를 섞는것이 일반적으로 좋다.

validation_data = data[:num_validation_samples]  # 검증 셋을 만든다.
training_data = data[num_validation_samples:]  # 훈련 셋을 만든다.
```



- `Scikit-learn`을 이용한 홀드아웃 검증 구현은 다음과 같다.

```python
from sklearn.model_selection import train_test_split

train_dat, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.33, random_state=42)
```



- 홀드아웃의 단점은 데이터가 적을 때, 검증 세트와 테스트 세트의 샘플이 너무 적어 주어진 전체 데이터를 통계적으로 대표하지 못할 수 있다.



#### K-Fold Cross Validation

- 데이터를 동일한 크기를 가진 `K`개의 분할로 나눈다.
- 각 분할 `i`에 대해 남은 `K-1`개의 분할로 모델을 훈련하고 분할 `i`에 대해 모델을 평가한다.
- 최종 점수는 `K`개의 점수의 평균을 구한다.
- 모델의 성능이 데이터 분할에 때라 편차가 클 때 유용하다.
- 모델의 튜닝에 별개의 검증 세트를 사용하게 된다.

![k-fold_cross_validation](./images/k-fold.PNG)



- K-fold cross validation의 구현 코드는 아래와 같다.

```python
k = 4
num_validation_samples = len(data) // k

np.random.shuffle(data)

validation_scores = []
for fold in range(k):
    # 검증 데이터 부분을 선택한다.
    validation_data = data[num_validation_samples * fold : num_validation_samples * (fold + 1)]
    # 남은 데이터를 훈련 데이터로 사용한다. 
    # 리스트에서 + 연산자는 두 리스트를 연결한다.
    training_data = data[:num_validation_samples * fold] + 
    	data[num+validation_samples * (fold + 1):]
    
    model = get_model()  # 훈련되지 않은 새로운 모델을 만든다.
    model.train(training_data)
    validation_score = model.evaluate(validation_data)
    validation_scores.append(validation_score)
    
validation_score = np.average(validation_scores)  # 검증 점수: K개의 폴드에 대한 평균

model = get_model()  
model.train(data)  # 테스트 데이터를 제외한 전체 데이터로 최종 모델을 훈련
test_score = model.evaluate(test_data)
```



- k-fold cross validation은 `scikit-learn`의 `cross_validate()`함수를 이용해 쉽게 구현할 수 있는데, 케라스 모델을 사이킷런과 호환되도록 `KerasClassifier`나 `KerasRegressor` 클래스로 모델을 감싸줘야 한다.

```python
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_validate

model = KerasClassifier(build_fn=get_model, epochs=150, batch_size=128, verbose=0)
kfold = cross_validate(model, data, labels, cv=4)
```



#### Iterated K-fold Validation with Suffling

- K-fold 검증을 여러 번 적용하되, `K`개의 분할로 나누기 전에 매번 데이터를 랜덤하게 섞는다.
- `P x K`개(P는 반복 횟수)의 모델을 훈련하고 평가하기 때문에 비용이 매우 많이 든다.
- 최종 점수는 모든 K-fold 검증(`P x K`)의 평균이다.

- 데이터가 적고 가능한 정확하게 모델을 평가하고자 할 때 사용한다.
- Kaggle에서는 이 방법이 매우 도움이 된다고 한다.
- 이 방법에 대한 예제는 해당 링크에서 확인할 수 있다. → [[링크](https://tensorflow.blog/2017/12/27/%EB%B0%98%EB%B3%B5-%EA%B5%90%EC%B0%A8-%EA%B2%80%EC%A6%9D/)]



### 4.2.2 유의해야 할 것

1. **대표성 있는 데이터** : 훈련 세트와 테스트 세트로 나누기 전에 데이터를 랜덤하게 섞어 주는 것이 좋다.
2. **시간의 방향** : 과거로 부터 미래를 예측해야 하는 시계열 데이터의 경우에는 데이터를 분할하기 전에 랜덤하게 섞어서는 절대로 안된다. 따라서, 테스트 데이터는 훈련 데이터보다 미래의 것이어야 한다.
3. **데이터 중복** : 훈련 세트와 검증 세트가 중복되지 않아야 한다.