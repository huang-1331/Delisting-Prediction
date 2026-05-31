# Methodology

## 연구 질문

재무제표 기반 머신러닝 모델은 기존 Altman Z-Score보다 상장폐지 예측 성능이 우수한가?

## 예측 단위

- 관측 단위: 기업-시점
- 입력 시점: `T`
- 예측 대상: `T+12개월` 이내 상장폐지 여부
- 타겟 변수: `상장폐지_t12`

## 데이터 누수 방지

모델 입력 변수는 `T` 시점에 관측 가능한 재무제표 정보로 제한한다.

TODO:

- 재무제표 공시일 기준으로 실제 이용 가능 시점 반영 여부 결정
- 사업보고서, 반기보고서, 분기보고서 중 사용할 보고서 유형 결정
- 상장폐지 일자와 관측 시점 간의 시간차 처리 규칙 확정

## 모델 비교 순서

1. Altman Z-Score
2. Logistic Regression
3. Random Forest
4. XGBoost

## 평가 지표

Accuracy는 주요 지표로 사용하지 않는다.

우선순위:

1. Recall
2. F1 Score
3. PR-AUC
4. Precision
5. Accuracy

필수 산출물:

- Confusion Matrix
- Recall
- Precision
- F1 Score
- ROC-AUC
- PR-AUC

