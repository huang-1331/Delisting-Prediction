# KOSDAQ 제조업 상장폐지 예측 모델 개발

## 프로젝트 개요

본 프로젝트는 KOSDAQ 제조업 상장기업의 재무제표 데이터를 활용하여 특정 시점 `T`의 정보만으로 `T+12개월` 이내 상장폐지 여부를 예측하는 머신러닝 모델을 개발한다.

학부 수준 연구 프로젝트이지만, 데이터 수집부터 라벨링, 특성공학, 모델링, 평가, 문서화까지 실제 산업 프로젝트에 가까운 재현 가능한 구조를 목표로 한다.

## 연구 질문

> 재무제표 기반 머신러닝 모델은 기존 Altman Z-Score보다 상장폐지 예측 성능이 우수한가?

## 연구 범위

2026-10-28 결정: 주 비교 기준은 장부자본을 사용하는 **Altman Z′**이며
위험 경계는1.23이다. 비상장 제조업용 모형을 상장기업에 적용하는 한계를
명시한다. 현재는 설정만 반영했고 기존 원형 계산을 Z′로 전환하는 구현과
본수집·모델 학습은 아직 수행하지 않았다.

- 대상 시장: KOSDAQ
- 대상 산업: 제조업
- 데이터: OpenDART 재무제표 데이터, 상장폐지 여부 데이터
- 예측 타겟: `상장폐지_t12`
- 예측 정의: `T` 시점 이후 12개월 이내 상장폐지 발생 여부
- 제외 범위: 딥러닝, 뉴스 텍스트 분석, 공시 원문 NLP, 비정형 데이터

## 데이터 출처

- OpenDART: 기업코드, 공시 접수일, 사업보고서 재무제표
- KIND: 현재 KOSDAQ 상장법인 목록과 KRX 업종명
- KRX Data Marketplace 화면 20037: 과거 상장폐지일과 상장폐지 사유
- 기업 universe: KOSDAQ 제조업 보통주 기업. 현재·과거 상장폐지기업 모두
  예측시점 T에 확인 가능했던 주된 업종으로 판정한다. KIND/KRX 업종명은
  후보 탐색용이며, 과거 보고서·공시로 확인하는 절차를 채택했다.
  실제 시점별 업종 복원과 분류체계 대응 검증은 미완료다.
- 연구기간: 2015~2024 사업연도

## 개발 일정

| 기간 | 주요 작업 |
| --- | --- |
| 2026-07 | 데이터 수집 자동화, OpenDART 연결 검증, 기업 universe 정의 |
| 2026-08 | 상장폐지 라벨링 로직 작성, T+12개월 타겟 정의 검증 |
| 2026-09 | 데이터 수집 완료, 특성공학·전처리, Altman Z-Score 구현 |
| 2026-10 | Logistic Regression, Random Forest 모델 개발 |
| 2026-11 | XGBoost 개발, 전체 모델 비교, 오류 분석 |
| 2026-12 | 최종 보고서 작성, 재현성 점검, 발표 자료 정리 |

2026-08-01에 선행 데이터 작업을 우선하도록 일정을 조정했으며, 후속 작업을
압축하여 최종 완료 기한을 2026년 12월로 확정하였다.

현재 목표, 실제 수집 결과, 진행도, 중단 사유와 향후 작업은
[`docs/project_status.md`](docs/project_status.md)에서 확인할 수 있다.

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

OpenDART 및 KRX 인증정보는 프로젝트 루트의 `.secrets.toml`에 설정한다.
이 파일은 Git 추적에서 제외되며, 형식은 `.secrets.toml.example`을 따른다.

```toml
[opendart]
api_key = "your-opendart-api-key"

[krx]
id = "your-krx-id"
password = "your-krx-password"
```

실제 공식자료 수집과 보고서용 터미널 기록은 다음 명령으로 실행한다. 중간에
중단되더라도 체크포인트를 기준으로 완료된 회사·연도는 자동으로 건너뛴다.

2026-10-28 갱신: Task02에서 업종 미확정 후보 보존과 안전한 재수집을 구현·검증했다.
기존 재무 CSV의 열 구조 불일치143행은 사용자 B안에 따라 원본 그대로 보존하며,
과거 성공 상태를 승계하지 않고 새 저장소에서 재수집한다.
아래 기존 명령은 과거 실행 예시이며 그대로 재개하지 않는다.
현재 옵션은 `python -m src.collection_runner --help`로 확인한다.
`--source-date`는 기존 공식자료 스냅샷 재사용, `--state`는 날짜와 독립된 요청 상태,
`--max-requests`, `--stock-codes`, `--years`는 소규모 검증 범위를 지정한다.
단위 검사·실제 후보군 검증은 `python -m src.task02_smoke`이며 실제 API 시도는
`--live`로 구분한다. 22개 검사와 실제 과거 폐지기업2개의 재무410행·공시108행,
재시작 추가요청0을 확인했다. 전체 수집·제조업 확정·모델 학습 완료는 아니다.
기본 상태는 `data/interim/collection_recollection_v2.sqlite3`이며 기존 파일은 보존한다.
검증 범위·실행 기록·한계는 `docs/task02_preparation.md`에 기록했다.

```bash
python -m src.collection_runner --stage all --as-of 2026-08-01
```

진행 기록은 실행 중 표준출력·표준오류와 동시에 실행 시각이 포함된
`reports/monthly/collection_runner_YYYYMMDD_HHMMSS_ffffff.txt`에 직접
저장된다. 보고서 증빙에는 이 직접 실행 로그만 사용한다.

API 키와 로그인 정보는 코드, 노트북, 문서에 직접 기록하지 않는다.

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
