# KOSDAQ 제조업 상장폐지 예측 모델 개발

## 프로젝트 개요

본 프로젝트는 KOSDAQ 제조업 상장기업의 재무제표 데이터를 활용하여 특정 시점 `T`의 정보만으로 `T+12개월` 이내 상장폐지 여부를 예측하는 머신러닝 모델을 개발한다.

학부 수준 연구 프로젝트이지만, 데이터 수집부터 라벨링, 특성공학, 모델링, 평가, 문서화까지 실제 산업 프로젝트에 가까운 재현 가능한 구조를 목표로 한다.

## 연구 질문

> 재무제표 기반 머신러닝 모델은 기존 Altman Z-Score보다 상장폐지 예측 성능이 우수한가?

## 연구 범위

- 대상 시장: KOSDAQ
- 대상 산업: 제조업
- 데이터: OpenDART 재무제표 데이터, 상장폐지 여부 데이터
- 예측 타겟: `상장폐지_t12`
- 예측 정의: `T` 시점 이후 12개월 이내 상장폐지 발생 여부
- 제외 범위: 딥러닝, 뉴스 텍스트 분석, 공시 원문 NLP, 비정형 데이터

## 데이터 출처

- OpenDART: 기업 재무제표 및 공시 기반 재무 데이터
- FinanceDataReader: 상장기업 목록, 시장 데이터 보조 수집
- TODO: KRX 또는 기타 공식 출처 기반 상장폐지 일자 데이터 확보 방식 확정
- TODO: KOSDAQ 제조업 분류 기준과 기준일 확정

## 개발 일정

| 기간 | 주요 작업 |
| --- | --- |
| 2026-06 | 데이터 수집 자동화, OpenDART 연결 검증, 기업 universe 정의 |
| 2026-07 | 상장폐지 라벨링 로직 작성, T+12개월 타겟 정의 검증 |
| 2026-08 | 재무비율 특성공학, 결측치 처리, 데이터 누수 점검 |
| 2026-09 | Altman Z-Score baseline 구현 및 평가 |
| 2026-10 | Logistic Regression, Random Forest 모델 개발 |
| 2026-11 | XGBoost 모델 개발, 모델 비교, 오류 분석 |
| 2026-12 | 최종 보고서 작성, 재현성 점검, 발표 자료 정리 |

## 사용 기술

- Python
- pandas, numpy
- scikit-learn
- xgboost
- OpenDartReader
- FinanceDataReader
- matplotlib, seaborn
- Jupyter Notebook

## 설치 및 실행 준비

```bash
pip install -r requirements.txt
```

OpenDART 수집 코드를 실행하기 전에 환경변수로 API 키를 설정한다.

```bash
export DART_API_KEY="your-dart-api-key"
```

API 키는 코드, 노트북, 문서에 직접 기록하지 않는다.

## 평가 원칙

Accuracy는 주요 지표로 사용하지 않는다. 성능 해석의 우선순위는 다음과 같다.

1. Recall
2. F1 Score
3. PR-AUC
4. Precision
5. Accuracy

모든 모델은 다음 지표를 계산한다.

- Recall
- Precision
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

## 모델 개발 순서

1. Altman Z-Score
2. Logistic Regression
3. Random Forest
4. XGBoost

## 데이터 누수 방지

모든 예측은 반드시 다음 구조를 따른다.

```text
T 시점 데이터
-> 모델 입력
-> T+12개월 상장폐지 여부
```

`T` 이후에 알 수 있는 재무정보, 시장정보, 상장폐지 결과 정보는 입력 변수로 사용하지 않는다.
