Holdout



## 시계열 데이터의 홀드아웃 검증

- 테스트 데이터에 가장 가까운 기간의 데이터를 검증 데이터로 삼음으로써 테스트 데이터에 대한 예측 성능이 높아질 것
  - -> 시점이 가까운 데이터일수록 우리가 예측할 데이터의 경향을 더 가깝게 반영한다는 가정에 기반을 둔 것
- 1년 단위로 주기성이 강한 데이터일 때는 가장 최근 데이터보다 테스트 데이터의 1년 전 기간을 검증 데이터로 삼을 수도 있음



``` python
# 변수 period를 기준으로 분할 (0~2 학습 데이터, 3 테스트 데이터)
# (3을 검증 데이터로 함)

is_tr = train_x['period'] < 3
is_va = train_x['period'] == 3
tr_x, va_x = train_x[is_tr], train_x[is_va]
tr_y, va_y = train_y[is_tr], train_y[is_va]
```



### 교차검증

308p

1. 학습 데이터 기간을 처음 시작점부터 잡을 경우
2. 학습 데이터 기간의 길이를 서로 맞추는 경우
3. ekstnsgl tlrksdmfh qnsgkf

```python
va_period_list = [1,2,3]
for va_period in va_period_list:
    is_tr = train_x['period'] < va_period
    is_va = train_x['period'] == va_period
    tr_x, va_x = train_x[is_tr], train_x[is_va]
    tr_y, va_y = train_y[is_tr], train_y[is_va]
    
tss = TimeSeriesSplit(n_split = 4)
for tr_idx, va_idx in tss.split(train_x):
    tr_x, va_x = train_x.iloc[tr.idx], train_x.iloc[va.idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
```

```python
# 변수 period가 0, 1, 2, 3인 데이터를 각각 검증 데이터로 하고
# 그 이외의 학습 데이터를 학습에 사용

va_period_list = [0,1,2,3]
for va_period in va_period_list:
    is_tr = train_x['period'] != va_period
    is_va = train_x['period'] == va_period
    tr_x, va_x = train_x[is_tr], train_x[is_va]
    tr_y, va_y = train_y[is_tr], train_y[is_va]
```



- 예측값과 실제값을 플롯하여 예측 성능이 안정되었는지 확인하거나, 검증 점수와 Public Leaderboard의 상관관계를 보면서 어디까지 검증 기간을 참고할 것인지, 학습 데이터 기간은 어디까지 제어할지 고려



