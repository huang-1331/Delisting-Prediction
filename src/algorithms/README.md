# 알고리즘 1·2 — 사용자 데이터 실행 코드

연구 데이터·기존 모델 계수·성능 결과는 포함하지 않습니다. 저장소 루트에서 실행하세요. 각 예측 기간은 해당 기간에 맞는 라벨 CSV로 따로 학습합니다.

| 구분 | 구현 | 학습 입력 |
|---|---|---|
| [알고리즘 1](algorithm_1/core.py) | 원래 PAIR-Guard: 10변수 기반 21개 변수 쌍·내부 paired CV·전문가 결합 | 10개 비율, 기업당 1행, 정상/사건 1:1 `pair_id` |
| [알고리즘 2](algorithm_2/core.py) | 유지한 V15: 6개 변수 구성·5개 가중치·3개 C 비교, 검증 Recall 하한에서 Precision 최대 선택 | 50개 후보 비율. 반복 기업·시점 허용 |

알고리즘 2는 실패/미채택 실험 V16·V17로 교체하지 않았습니다. 기존 학습 계수 없이 사용자 자료로 다시 학습합니다. 알고리즘 1의 `core.py`는 원본 Git blob `b531509532b80dbe902827789b1e1b0c2a5eadf2`와 동일합니다. CLI·정보일자 검사는 별도 어댑터입니다.

## 설치

```bash
python -m pip install -r src/algorithms/requirements.txt
```

## CSV 준비

```bash
python -m src.algorithms.algorithm_1 template --output inputs/algorithm1.csv --target-column 상장폐지_t12
python -m src.algorithms.algorithm_2 template --output inputs/algorithm2.csv --target-column 상장폐지_t12
```

헤더만 있는 템플릿이 생성됩니다. 실제 자료로 채우고 기업이 겹치지 않는 `train.csv`, `validation.csv`, `test.csv`를 준비합니다. 알고리즘 1은 같은 쌍 전체를 같은 분할에 넣어야 합니다. 학습에는 최소 4개 완전한 쌍이 필요합니다. 최소치가 신뢰할 만한 연구 표본 수라는 뜻은 아닙니다.

- 공통 필수: `corp_code`, `anchor`, `available_at`, `known_procedure_date` 및 금융 비율. 알고리즘 1은 `pair_id`도 필수입니다.
- `available_at`은 입력에 사용한 모든 공시 중 가장 늦은 **실제 공개일**입니다. 입력이 아직 공개되지 않았거나 절차가 이미 알려진 행은 제외합니다. 이후의 절차 공시로 과거 행을 소급 제외하지 않습니다.
- 날짜는 일관된 ISO 8601로 작성하세요. 시각을 사용하면 시간대(`+09:00` 등)를 명시합니다. 날짜만 같게 입력한 공개일은 기준일에 이미 공개된 것으로 처리됩니다.
- 비율은 소수 단위입니다(50%는 `0.5`). TTM은 최근 4개 연속 분기 합계입니다. 미래 분기·미래 상장폐지 정보를 변수 계산에 포함하면 안 됩니다.
- 입력은 **공학 처리한 재무비율**입니다. DART 원본 계정 CSV를 그대로 넣는 수집기/라벨 생성기는 아닙니다. [입력 열·계산식](input_schema.json)을 확인하세요.
- **TODO:** 실제 상장폐지 라벨 출처, KOSDAQ 제조업 기준일, 관측 가능 공시 빈티지, 연속 추적·우측 검열 처리는 데이터 작성자가 확인해야 합니다. `상장폐지_t12`를 자본잠식 여부로 임의 대체하지 않습니다.

## 학습

```bash
python -m src.algorithms.algorithm_1 train --train inputs/train1.csv --validation inputs/validation1.csv --target-column 상장폐지_t12 --horizon-months 12 --model outputs/algorithm1.json
python -m src.algorithms.algorithm_2 train --train inputs/train2.csv --validation inputs/validation2.csv --target-column 상장폐지_t12 --horizon-months 12 --minimum-recall 0.5 --model outputs/algorithm2.json
```

알고리즘 1은 원래의 학습 중앙값·1/99% 경계 제한·표준화와 검증 G-mean 규칙을 유지합니다. 알고리즘 2는 학습 중앙값·Z-score만 적용하고 극단치 행을 삭제하거나 원비율을 경계로 제한하지 않습니다. 5개 파생변수의 `max`는 변수 정의입니다. 평균·결측 보완·경계·가중치와 변수 선택에 test는 사용하지 않습니다.

알고리즘 1에서 정보일자 검사로 한 쌍의 일부가 제외되면 그 쌍 전체를 학습에서 제외합니다. 두 알고리즘 모두 부실/정상 두 클래스가 학습·검증 자료에 필요합니다.

## 새 데이터 예측 / 고정 테스트 평가

```bash
python -m src.algorithms.algorithm_1 predict --model outputs/algorithm1.json --input inputs/new1.csv --output outputs/predictions1.csv
python -m src.algorithms.algorithm_2 predict --model outputs/algorithm2.json --input inputs/new2.csv --output outputs/predictions2.csv
python -m src.algorithms.algorithm_1 evaluate --model outputs/algorithm1.json --input inputs/test1.csv --output outputs/metrics1.json
python -m src.algorithms.algorithm_2 evaluate --model outputs/algorithm2.json --input inputs/test2.csv --output outputs/metrics2.json
```

`predict`에는 정답 열이 없어도 됩니다. `evaluate`에는 학습에서 지정한 정답 열이 필요하며 학습·검증 기업과 겹치면 거부합니다. 평가 결과는 Recall·F1·PR-AUC·Precision·ROC-AUC·혼동행렬과 참고 정확도를 제공합니다. 예측은 입력과 같은 행 수를 유지하고, 정보일자/결측 조건 때문에 평가 불가인 행을 상태와 빈 점수로 표시합니다.

선택한 모델의 필요한 입력 열만 `predict`에 제공할 수 있습니다. 날짜 메타데이터는 항상 필요합니다. `risk_score`는 0~1의 **보정되지 않은 순위 점수**이며 실제 상장폐지 확률이 아닙니다. 검증 Recall 하한을 충족해도 test·미래 Recall은 보장되지 않습니다. 기존 연구의 성능 수치는 이 코드로 새 데이터에서 얻을 성능을 보장하지 않습니다.
