# 풍력발전기 출력 추정: 물리식과 머신러닝 비교

소프트웨어랩(EE370) Term Project · 김사무엘

GitHub: https://github.com/samueli02/wind-turbine-power-physics-ml

## 주제와 강의 연계

풍속과 풍향이 주어졌을 때 같은 시점의 풍력발전기 출력을 추정하고 실제 계측값과 비교한다. **미래 풍속 예보**가 아니라 측정된 입력에 대한 출력 추정이다.

- **4강 행렬·최소제곱법(LSM):** 제조사 이론 곡선에 대해 `P = k v³`의 계수 `k = Σ(v³P)/Σ(v⁶)`를 학습 기간에만 적합한다.
- **7강 머신러닝:** 풍속, 풍향의 `sin`·`cos`를 입력으로 랜덤 포레스트 회귀 모델을 학습한다.
- 물리식의 출발점은 `P_wind = ½ρAv³`, 터빈 출력은 `P_turbine = ½ρAC_pηv³`이다. 이 데이터에는 `ρ`, `A`, `C_p`, `η`가 없으므로 이들의 합성 계수 `k`를 제조사 곡선에서 추정한다. 시동 풍속 미만에서 0, 정격 출력에서 상한을 둔다. 이것은 **단순화한 물리 기준선**이지 실제 터빈 작동을 완벽히 재현하는 식이 아니다.

## 개발 환경과 실행

Windows 11, Python 3.12에서 검증했다. 검증 버전: NumPy 2.3.5, pandas 3.0.1, scikit-learn 1.9.1, matplotlib 3.11.2. Python 3.10 이상을 권장한다.

프로젝트 루트에서 실행:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python src\main.py --data data\T1.csv --out results
```

macOS/Linux에서는 마지막 두 줄의 `.venv\Scripts\python`을 `.venv/bin/python`으로 바꾼다. 저장소만 내려받았다면 [Kaggle 원본](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset)에서 `T1.csv`를 받아 `data/T1.csv`로 놓는다. 제출용 ZIP에는 데이터가 포함된다.

## 방법과 결과

원본 50,530행 중 음수 실측 출력 57행을 제외했다. 유효 50,473행을 시간순으로 정렬하고 앞 75%(37,854행, 2018-09-26 19:10까지)를 학습, 뒤 25%(12,619행)를 평가에 썼다. 물리 계수와 ML 모두 평가 기간을 학습에 사용하지 않는다. 제조사 곡선은 별도의 참고 기준이다.

| 방법 | MAE (kW) | RMSE (kW) | 평균 오차, 예측−실측 (kW) | R² |
|---|---:|---:|---:|---:|
| 단순 물리식 (`kv³`, 시동/정격 반영) | 210.000 | 450.208 | +180.197 | 0.8837 |
| 제조사 제공 이론 곡선 | 192.641 | 429.356 | +173.981 | 0.8942 |
| 랜덤 포레스트 | 236.483 | 480.800 | −97.351 | 0.8674 |

R²는 같은 평가 기간의 실측 출력 평균을 기준으로 `1 − Σ(실측−예측)² / Σ(실측−평균 실측)²`로 계산한다. 1에 가까울수록 실측 변동을 잘 설명한다.

**결론:** 전체 평가 기간에는 머신러닝이 단순 물리식보다 정확하지 않았다. 풍속 0–12 m/s 구간에서는 ML의 MAE가 물리식보다 낮았지만, 12 m/s 이상에서 ML 오차가 크게 증가했다. `results/03_mae_by_wind_speed.png`와 `mae_by_wind_speed.csv`에 구간별 결과가 있다. 원인은 이 데이터만으로 단정할 수 없다. 실제 출력 0이면서 풍속이 12 m/s 이상인 평가 사례가 35개 있고, 정지·제한 운전 상태를 나타내는 변수가 없다는 한계가 있다. 또한 제조사 곡선과 물리식의 양의 편향은 실제 운전 출력을 과대평가한다는 뜻이다.

`results/01_time_comparison.png`는 평가 첫 24시간, `02_power_curve.png`는 풍속별 출력, `03_mae_by_wind_speed.png`는 풍속 구간별 MAE다. `predictions.csv`에는 각 평가 시점의 입력·실측·세 예측이 있고 `metrics.json`에는 학습 구간과 지표가 있다. 실행할 때 이 파일들이 다시 생성된다.

## 한계와 후속 실험

- 같은 시점 풍속과 풍향을 알고 있다는 전제라서 장래 발전량 예보로 해석할 수 없다.
- 물리식의 `k`는 제조사 곡선에서 추정한 합성 계수이며 공기 밀도나 블레이드 반지름을 독립적으로 검증한 값이 아니다.
- 운전 상태, 날개 피치, 기온·기압 데이터가 없어 고풍속 저출력 사례의 이유를 확인할 수 없다.
- 후속 실험은 월별 시계열 검증, 풍속 구간별 모델, 운전 상태 변수가 있는 별도 데이터로 진행할 수 있다. 이들은 이번 결과에서 검증하지 않았다.

## AI 활용 내역

OpenAI ChatGPT/Codex를 주제 탐색, 데이터 출처 조사, 코드·문서·발표자료 초안 작성에 사용했다. 코드 실행으로 행 수, 시간순 분할, 결과 파일 생성, 오차 지표를 확인했다. 물리식과 데이터 단위는 아래 출처와 대조했다. **제출자는 코드와 수식·그래프를 직접 이해하고 발표해야 한다.**

## 참고자료

1. 소프트웨어랩 Term Project 안내자료(4강 행렬·LSM, 7강 ML).
2. [Kaggle, Wind Turbine Scada Dataset](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset).
3. [U.S. Department of Energy, Small Wind Guidebook](https://www.energy.gov/cmei/systems/windexchange/small-wind-guidebook), 풍력 `P = ½ρAC_pv³` 관계.
4. [Investigation of the Features Influencing the Accuracy of Wind Turbine Power Calculation at Short-Term Intervals](https://www.mdpi.com/2571-5577/7/6/105), 2018 SCADA와 이론 곡선 단위 참고.
