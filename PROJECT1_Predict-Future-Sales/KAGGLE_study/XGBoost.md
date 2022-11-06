## 모델구축



### 모델 학습과 예측

1) 모델 종류를 선택하고 하이퍼파라미터를 지정
2) 학습 데이터와 목적변수를 제공하여 학습 진행
3) 테스트 데이터를 제공하여 예측



```python
# 모델의 하이퍼파라미터를 지정
params = {'param1': 10, 'param2': 100}

# Model 클래스를 정의
# Model 클래스는 fit로 학습하고 predict로 예측값 확률을 출력

# 모델 정의
model = Model(params)

# 학습 데이터로 모델 학습
model.fit(train_x, train_y)

# 테스트 데이터에 대해 예측 결과를 출력
pred = model.predict(test_x)
```



### 모델 검증

- validation : 학습이 진행되면서 학습 데이터에 대한 점수와 검증 데이터에 대한 점수가 어떻게 달라지는지를 모니터링할 수 있음

```python
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# 학습 데이터 검증 데이터를 나누는 인덱스를 작성
# 학습 데이터를 4개로 나누고 그중 하나를 검증 데이터로 지정
kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]

# 학습 데이터와 검증 데이터로 구분
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# 모델 정의
model = Model(params)

# 학습 데이터에 이용하여 모델 학습 수행
# 모델에 따라서는 검증 데이터를 동시에 제공하여 점수 모니터링
model.fit(tr_x, tr_y)

# 검증 데이터에 대해 예측하고 평가 수행
va_pred = model.predict(va_x)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')
```



- 홀드아웃(Hold-out) : 일부 데이터를 검증용으로 나누는 방법

<br>

- 교차검증(cross validation) : 홀드아웃으로 모델의 학습이나 평가에 사용할 수 있는 데이터가 그만큼 줄어드는 만큼, 효율적인 데이터 사용을 위해 사용
  - 1. 학습 데이터를 여러 개로 분할(폴드(fold))
    2. 그중 하나를 검증 데이터, 나머지를 학습 데이터로 삼아 학습 및 평가를 실시하고 검증 데이터에서의 점수를 구함
    3. 분할한 횟수만큼 검증 데이터를 바꿔가며 2의 내용을 반복하여 점수를 구함
    4. 검증 데이터의 평균 점수로 모델의 좋고 나쁨을 평가



```python
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# 학습 데이터를 4개로 나누고 그중 1개를 검증 데이터로 지정
# 분할한 검증 데이터를 바꾸어가며 학습 및 평가를 4회 실시
scores = []
kf = KFold(n_splits=4, shuffle=True, random_state=71)

for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    model = Model(params)
    model.fit(tr_x, tr_y)
    va_pred = model.predict(va_x)
    score = log_loss(va_y, va_pred)
    scores.append(score)
    
# 교차 검증의 평균 점수를 출력
print(f'logloss: {np.mean(scores):.4f}')
```



### 더 좋은 모델을 찾기

- 특징의 추가 및 변경
- 하이퍼파라미터 변경
- 모델 종류 변경
  - GBDT, 신경망, 앙상블



### 모델 선택 방법

- 최우선 선택 모델 : GBDT
- 문제에 따라 2순위로 검토, 다양성 : 신경망, 선형 모델
- 다양성 요구 : K-최근접 이웃 알고리즘, 랜덤 포레스트(RF)/ERT, RGF, FFM



## GBDT

그레이디언트 부스팅 결정 트리 (gradient boosting decision tree) (GBDT)

1.  목적변수와 예측값으로부터 계산되는 목적함수를 개선하고자 결정 트리를 작성하여 모델에 추가

2. 하이퍼파라미터에서 정한 결정 트리의 개수만큼 1 반복

순으로 진행되다가 각 결정 트리의 분기 및 잎의 가중치가 정해짐



트리를 작성하는 동안 모델의 예측값이 목적변수의 실젯값에 가까워짐 >> 작성되는 결정 트리의 가중치는 차츰 작아짐



최종 예측값 : 예측 대상 데이터를 각각 결정 트리에서 예측한 결과를 합산한 결과
$$
예측값\ y=\sum_{m=1}^M W_m
$$


> > 특징

- 특징은 수치로 표현해야 함
- 결측값을 다룰 수 있음
- 변수 간 상호작용이 반영



---------------

## XGBoost

```python
import xgboost as xgb
from sklearn.metrics import log_loss

# 특징과 목적변수를 xgboost의 데이터 구조로 변환
dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)
dtest = xgb.DMatrix(test_x)

# 하이퍼파라미터 설정
params = {'objective': 'binary:logistic','verbosity':0, 'random_state':71}
num_round = 50

# 학습의 실행
# 검증 데이터도 모델에 제공하여 학습 진행과 함께 점수가 어덯게 달라지는지 모니터링
# watchlist로 학습 데이터 및 검증 데이터 준비
watchlist = [(dtrain, 'train'), (dvalid, 'evel')]
model = xgb.train(params, dtrain, num_round, evals=watchlist)

# 검증 데이터의 점수를 확인
va_pred = model.predict(dvalid)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')

# 예측 : 두 값(0 or 1)의 예측이 아닌 1일 확률을 출력
pred = model.predict(dtest)
```



### booster

- gbtree (default)
- gblinear : 선형 모델
- dart : 정규화에 DART라는 알고리즘을 사용한 GBDT



### 목적함수

- 회귀 : `reg:squarederror`(평균제곱오차를 최소화하도록 학습)
- 이진 분류 : `binary:logistic` (로그 손실을 최소화하도록 학습)
- 다중 클래스 분류 : `multi:softprob` (다중 클래스 로그 손실을 최소화하도록 학습)



### Hyperparameter

- eta,learning_rate : Learning Rate (0~1)
  - (학습률 (0~1))
- n_estimators : Number of Weak Learner (ex. Decision Tree), (default: 10)
  - (약한 학습기의 개수)
- min_child_weight : Minimum sum of weights for all observations needed in child. (default: 1)
  - (child 에 필요한 모든 관측지에 대한 관측치에 대한 가중치의 최소 합)
- max_depth : Tree max_depth (default: 6)
  - (트리의 깊이 낮으면 낮을수록 과적합을 방지합니다.)
- subsample : Data Sampling rate for tree
  - (각 트리별 데이터 샘플링 비율)
- colsample_bytree : feature sampling rate for tree
  - (각 트리별 feature 샘플링 비율)
- gamma : It is the minmum loss reduction value that will determine the further division of leaf node
  - (리프노드의 추가분할을 결정할 최소손실 값이다.)
- reg_lambda : L2 Regulation
  - (L2 가중치)
- reg_alpha : L1 Regulation
  - (L1 가중치)
- scale_pos_weight : Balancing unbalanced datasets
  - (불균형 데이터셋의 균형 유지)





### 학습 데이터와 검증 데이터의 점수 모니터링

```python
# 모니터링을 logloss로 수행. early_stopping_rounds를 20라운드로 설정
params = {'objective':'binary:logistic','verbosity':0, 'random_state'=71, 'eval_metric':'logloss'}
num_round = 500
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params, dtrain, num_round, evals=watchlist,
                 early_stopping_rounds=20)

# 최적의 결정 트리의 개수로 예측
pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
```



### 알고리즘

행 데이터 개수 N, 결정 트리 M

1. 결정 트리에 따른 부스팅
   - m 번째 트리를 학습시킬 때는 m-1번째 트리까지의 예측 오차를 보정하도록 m 번째 트리 결정
   - 학습이 진행되면서 보정이 필요한 정도가 줄어들기 때문에 결정 트리의 가중치는 점차 작아짐
   - 결정트리 m 에서 , w_m(x_i) : 잎의 가중치

$$
\sum_{m=1}^M w_m(x_i)
$$

> > 1. 결정 트리를 M개 작성
> > 2. 결정 트리는 분기 작성을 반복. >> 어느 특징의 어떤 값으로 분기할지 선택
> > 3. 어느 특징의 어떤 값으로 분기할지는 모든 후보를 조사하고 분기시켜 최적의 잎의 가중치를 설정했을 때 **목적함수가 가장 크게 감소**하는 걸로 결정
> > 4. 결정 트리가 만들어지면 그에 기반하여 예측값 갱신



2. 정규화된 목적함수

   - $$
     l(y_i, \hat{y_i}) \\(y_i : 목적변수,\ \hat{y_i} : 예측값)
     $$

   - 각각의 결정 트리를 f_m이라고 했을 때, 결정 트리에 대해 벌칙이 계산되는 정규화항을 \Omega(f_m) 이라고 함

   $$
   L = \sum_{i=1}^N l(\hat{y_i},y_i)+ \sum_{m=1}^M \Omega(f_m)
   $$

   



3. 결정 트리 작성

   - 하나의 잎에서 시작해 분기를 반복함으로써 작성

   - 경사(그레이디언트(gradient))를 사용해 결정 트리를 어떻게 분기할지 결정

   - 뉴턴방법(Newton's method)처럼 이계도함수의 값도 이용

   - 목적함수 감소 계산

     - $$
       L_j = \sum_{i \in I_j}l(y_i, \hat{y_i}+w_j)
       $$

     - 각 행 데이터의 예측값의 주변 경사, 이계도함수

     - $$
       예측값 \hat{y_i}의 주변경사 g_i = \frac{\partial l}{\partial \hat{y_i}}, 이계도함수 h_i = \frac{\partial^2l}{\partial \hat{y_i}^2}
       $$

       라고 하면, 목적함수의 합은

     $$
     \tilde{L}_j = \sum_{i \in I_j}(l(y_i, \hat{y_i}) + g_iw_j + \frac{1}{2}h_iw^2_j)
     $$

     

     - l(y,y_hat)은 가중치를 정하는 과정에서 영향X

     - $$
       {\tilde{L}}'_j = \sum_{i \in I_j}( g_iw_j + \frac{1}{2}h_iw^2_j)
       $$

     - 목적함수 L을 최소로하는 가중치 w는

     - $$
       w_j = - \frac{\sum_{i \in I_j}g_i}{\sum_{i \in I_j}h_i},\\
       \tilde{L'}_j=-\frac{1}{2}\frac{(\sum_{i \in I_j}g_i)^2}{\sum_{i \in I_j}h_i}
       $$

     -  분기에 의한 목적함수의 감소

     - $$
       \tilde{L'}_j - (\tilde{L'}_{jL}+\tilde{L'}_{jR})
       $$

   



-----------

```python
# This is an example. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


cust_df = pd.read_csv('../input/santander-customer-satisfaction/train.csv')

cust_df['var3'].value_counts()
cust_df['var3'].replace(-999999, 2, inplace = True)
cust_df.drop('ID', axis = 1, inplace = True)

X_feature = cust_df.iloc[: , :-1]
y_label = cust_df.iloc[:,-1]

X_train, X_test, Y_train, Y_test = train_test_split(X_feature, y_label, test_size = 0.2, random_state = 56)
xgb_clf = XGBClassifier(n_estimators = 100, random_state = 156)
xgb_clf.fit(X_train, Y_train, early_stopping_rounds = 100,eval_metric ='auc', eval_set = [(X_test, Y_test)])  
```



- feature 중요도 (XGBoost Plot_imortance)

```python
from xgboost import plot_importance
import matplotlib.pyplot as plt
fig , ax = plt.subplots(1, 1, figsize = (10, 10))
plot_importance(xgb_clf, max_num_features= 20, ax = ax, height = 0.4)
```



- 의사결정트리 시각화 (XGBoost plot_tree)

```python
from xgboost import plot_tree
fig, ax = plt.subplots(figsize=(80, 80))
xgboost.plot_tree(xgb_clf, num_trees=4, ax=ax)
plt.show()
```

