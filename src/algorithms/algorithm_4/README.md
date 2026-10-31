# 알고리즘4: 자본부실 12·24개월 예측

예측 대상은 자본잠식률 `(1 - 자본총계 / 자본금) × 100`이 관측 이력에서 처음으로 **50%를 초과**하는 사건입니다. 예측 시점에는 기준 미해당이며, 이후 12개월 또는 24개월 안의 발생 여부를 예측합니다. 자본금은 양수여야 합니다. 상장폐지 확률 또는 완전자본잠식 확률을 출력하는 모델이 아닙니다.

## 구성과 실행

저장된 프로파일 로지스틱 전문가 모형과 XGBoost를 결합한 고정 앙상블입니다. 50개 재무 파생변수로부터 5개 상호작용을 추가하고 저장된 학습 중앙값·평균·표준편차로 변환합니다. 새 데이터로 학습·표준화 재적합·임계값 조정을 하지 않습니다.

```bash
pip install -r src/algorithms/algorithm_4/requirements.txt
python -m src.algorithms.algorithm_4 features50.csv predictions.csv --horizon 12
python -m src.algorithms.algorithm_4 features50.csv predictions24.csv --horizon 24
```

입력은 기업·예측시점당 한 행의 **표준화 전 50개 파생변수 CSV**입니다. 정확한 열 이름은 models/preprocessing_12.json 및 preprocessing_24.json의 columns 중 5개 상호작용 열을 제외한 이름입니다. core.py가 열 누락을 검사하며, 셀 결측은 저장된 학습 중앙값으로 보완합니다. 원 재무제표 10개 항목을 곧바로 넣는 인터페이스가 아니며, 분기 이력에서 50개 변수를 계산하는 수집·시계열 전처리 통합은 TODO입니다.

출력: horizon_months, research_score, warning, threshold와 입력에 있는 기업·시점 식별자. research_score는 검증된 확률이 아니라 연구용 점수입니다. models의 정책에 저장된 임계값을 그대로 적용합니다.

## 테스트 전용 조건과 한계

- 라벨 생성 시 예측 시점에 이미 파산·회생·상장폐지 절차가 공개된 기업은 제외합니다. 이 CLI는 사전 구성된 입력을 평가하므로 해당 절차·공시 공개일·상장 이력 검증은 입력 생성 단계에서 완료해야 합니다.
- 알고리즘4의 조합·임계값 선정에 일부 기존 평가 정답이 사용됐습니다. 저장 모델의 test_metrics는 탐색에 노출된 표본 성능이며 독립 검증 성능이 아닙니다.
- 독립 부실50·정상50 테스트셋은 미완료입니다. 새로운 테스트 자료는 학습·변수 선택·전처리 재적합·임계값 조정에 사용하면 안 됩니다.
- 최신 정정 재무값이 과거 예측 시점에도 공개됐다고 가정하면 안 됩니다. 실사용 전 point-in-time 입력과 독립 평가가 필요합니다.
- 이 저장소의 기존 상장폐지 연구와 별도의 자본부실 모델로 추가했습니다.

출처: ysookang028-crypto/financial-distress-prediction의 research/outputs/joint_constraints_v33. 저장된 정책·전처리·모델 JSON을 그대로 복사했으며 추론 함수만 학습 코드에서 분리했습니다. 원본 함수와 수치 동일성 검사는 진단 입력으로 수행하며 실제 독립 성능 검증을 의미하지 않습니다.
