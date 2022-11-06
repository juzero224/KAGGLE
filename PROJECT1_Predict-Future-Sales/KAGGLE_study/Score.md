# 평가지표



- 학습시킨 모델의 성능이나 그 예측 결과의 좋고 나쁨을 측정하는 지표



## 회귀의 평가지표

### RMSE

https://jysden.medium.com/%EC%96%B8%EC%A0%9C-mse-mae-rmse%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94%EA%B0%80-c473bd831c62

- RMSE (Mean Square Error)
    - 예측값과 정답의 차이를 제곱한 값의 평균을 구하고, 제곱근을 구함
    - 이상치에 민감하다는 MSE의 단점을 다소 보완
    - 각 오차가 다른 가중치를 갖음
    - 즉, 정답에 대해 예측값이 매우 다른 경우 그 차이는 오차값에 상대적으로 적게 반영
    - 미분 불가능한 지점을 찾는 것이 목표
    
$$
RMSE = \sqrt{\frac{1}{N}\sum_i^N(y_i - \hat{y}_i)^2}
$$

- N : 행 데이터의 수
- i = 1, 2, ..., N : 각 행 데이터의 인덱스
- y_i: i번째 행 데이터의 실젯값
- y_hat_i : i번째 행 데이터의 예측값



- RMSE의 값을 최소화했을 때의 결과가, 오차가 정규분포를 따른다는 전제하에 구할 수 있는 최대가능도방법과 같아짐
- 하나의 대푯값으로 예측을 실시한다고 했을 때 평가지표 RSME를 최소화하는 예측값 = 평균값
- MSE보다는 이상치에 덜 민감하지만 MAE 보다 이상치의 영향을 더 받음
- 이상치를 제외한 처리를 미리 해두지 않으면 과적합한 모델을 만들 가능성 있음



- metrics 모듈의 `mean_squared_error` 함수

```python
# RMSE
from sklearn.metrics import mean_squared_error

# y_true : 실젯값, y_pred : 예측값
y_true = [1.0, 1.5, 2.0, 1.2, 1.8]
y_pred = [0.8, 1.5, 1.8, 1.3, 3.0]

rmse = np.sqrt(mean_sqred_error(y_true, y_pred))
print(rmse)
# 0.5532
```

```python
def RMSE(pred,target,epochs=len(pred)):
    losses=[]
    for i in range(epochs):
        losses.append(np.sqrt(np.sum((pred[i]-target[i])**2)/len(pred)))
    return losses
```



### RMSLE

- RMSLE (root mean square logarithmic error)
  - 실젯값과 예측값의 로그를 각각 취한 후, 그 차의 제곱평균제곱근으로 계산
  - RMSE를 최소화 >> RMSLE 최소화


$$
RMSLE = \sqrt{\frac{1}{N}\sum_{i=1}^N(log(1+y_i)-log(1+\hat{y}_i))^2}
$$

- 예측할 대상의 목적변수를 정하고, 이 변수의 값에 로그를 취한 값을 새로운 목적 변수
- 목적변수의 분포가 한쪽으로 치우치면 큰 값의 영향력이 일반적인 RMSE보다 강해지기 때문에 이를 방지하려고 사용
- 실젯값과 예측값의 **비율**을 측정 지표로 사용하고 싶을 경우에 사용



- metrics 모듈의 `mean_squared_log_error` 함수

```python
rmsle = np.sqrt(mean_sqred_log_error(y_true, y_pred))
```



```python
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmsle(y, y0):
    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))
```







### MAE

- MAE (mean absolute error)
  - 실젯값과 예측값의 차에 대한 절댓값의 평균으로 계산


$$
MAE = \frac{1}{N}\sum_{i=1}^N|y_i-\hat{y}_i|
$$

- MAE는 이상치의 영향을 상대적으로 줄어주는 평가에 적절
- 이상치에 robust함. MSE, RMSE에 비해 이상치의 영향을 상대적으로 덜 받는다.
- 함수값에 미분 불가능한 지점이 있어 다루기가 어렵다
- 하나의 대푯값으로 예측을 할 때 MAE를 최소화하는 예측값 = 중앙값(median)



- metrics 모듈의 `mean_absolute_error` 함수

```python
def MAE(pred,target,epochs=len(pred)):
    losses=[]
    for i in range(epochs):
        losses.append(np.sum(np.abs(pred[i]-target[i]))/len(pred))
    return losses
```

```python
# MAE
from sklearn.metrics import mean_absolute_error

# y_true : 실젯값, y_pred : 예측값
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

MAE = mean_absolute_error(y_true, y_pred)
print(MAE)
# 0.75
```





### 결정계수

- 결정계수 (coefficient of determination) (R^2)
  - 회귀분석의 적합성
  - 분모는 예측값에 의존하지 않고, 분자는 오차의 제곱에 관한 것이므로 이 지표의 최대화 = RMSE의 최소화


$$
R^2 = 1- \frac{\sum_{i=1}^N(y_i-\hat{y}_i)^2}{\sum_{i=1}^N(y_i-\bar{y}_i)^2}
$$

$$
\bar{y} = \frac{1}{N}\sum_{i=1}^Ny_i
$$

- 결정계수이 최댓값 = 1
- 1에 가까울 수록 모델 성능이 높음



- metrics 모듈의 r2_score 함수





-------

참고

출처: https://investcommodity.tistory.com/entry/모델-평가지표 [ire:티스토리]