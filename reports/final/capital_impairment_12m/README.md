# 12개월 자본잠식 조기경보 최종 모델

고정된 v75 히스토그램 기반 그래디언트 부스팅 모델이다. 예측 대상은 예측 시점 이후 12개월 내 관측된 최초 자본잠식률 35% 초과다. 이 모델과 평가는 저장소의 상장폐지 타깃 `상장폐지_t12` 평가와 별도로 관리한다.

## 실행

저장소 루트에서 실행한다. Python 환경에 `scikit-learn==1.8.0`, pandas, numpy, joblib가 필요하다.

```bash
python src/capital_impairment_12m_final/predict.py features.csv warnings.csv
```

입력 CSV는 예측 시점까지 확보된 분기 재무자료로 계산한 특성 표다. 금액 비율은 소수로 입력한다. `corp_code`, `feature_end`, `anchor` 열이 있으면 출력에 전달한다. 결과는 위험점수와 경고 여부이며 경고 기준은 0.01397876000041635다.

| 입력 구분 | CSV 열 |
|---|---|
| 잠식·자기자본 | impairment_ratio, impairment_delta_1q, impairment_mean_4q, equity_delta_4q_assets, equity_assets |
| 부채·유동성 | debt_assets, debt_assets_delta_1q, current_ratio, working_capital_assets |
| 수익·손실 | roa_ttm, operating_roa_ttm, loss_fraction_4q, retained_assets, retained_deficit_to_capital |

`equity_assets`는 파생 계산에 쓰인다. 모델은 위 기본 입력에서 35% 기준 접근 압력, 두 분기 악화 지속 시 추정 잠식률, 자본금 대비 부채 및 손실 추정치를 계산해 총 17개 특성을 사용한다. 결측 대체 및 표준화에는 저장된 학습 통계량을 사용한다.

## 기업 단위 홀드아웃 검증

910개 보류 기업 중 입력 및 추적 조건을 충족한 770개에서 최근 적격 관측 한 건씩 평가했다. 저장된 모델과 경고 기준을 그대로 적용했다.

| 지표 | 결과 |
|---|---:|
| Recall | 79.0% (64/81) |
| 정상 오경보율 | 19.4% (134/689) |
| Precision | 32.3% (64/198) |
| F1 | 0.459 |
| ROC-AUC | 0.861 |
| PR-AUC (Average Precision) | 0.497 |
| 혼동행렬 | TP 64, FP 134, FN 17, TN 555 |

표본 지표는 Recall 70% 이상과 정상 오경보율 20% 이하를 충족했다. Precision 80% 목표에는 미달했다. Recall과 오경보율의 개별 95% 신뢰구간은 각각 68.9–86.5%, 16.7–22.6%다. 이 평가는 보류 기업의 기존 역사 자료를 사용했다. 상장 이력과 최초 공시 당시 값의 전수 검증은 별도 과제다.
