# 완전자본잠식 예측 모델 A B C

예측 기준일 이후 12개월 내 별도재무제표 자기자본이 처음 0 이하로 관측되는지를 예측합니다. 예측일에 공개된 재무자료에서 만든 56개 입력을 사용합니다. 모델과 경고선은 시점가용성 기준을 수정해 재평가한 실행에서 가져왔습니다.

| 모델 | 코드 | 학습 완료 파일 | 기업 F1 | Precision | Recall |
|---|---|---|---:|---:|---:|
| A 로지스틱 회귀 | `model_a.py` | `weights/model_a.joblib` | 0.2745 | 16.67% | 77.78% |
| B 랜덤 포리스트 | `model_b.py` | `weights/model_b.joblib` | 0.2319 | 13.33% | 88.89% |
| C XGBoost | `model_c.py` | `weights/model_c.joblib` | 0.2034 | 12.00% | 66.67% |

표의 지표는 기업이 겹치지 않는 테스트 207개 기업·6,406건에 대한 결과입니다. 이 중 완전자본잠식이 관측된 기업은 9개입니다. 입력 열, 경고선 및 모델 파일의 SHA-256은 `feature_contract.json`에 있습니다.

저장소 루트에서 의존성을 설치하고 입력 파일의 56개 재무 변수를 전달합니다.

```bash
python -m pip install -r src/models_abc/requirements.txt
```

```python
import pandas as pd
from src.models_abc import model_a
from src.models_abc.common import load_fitted

inputs = pd.read_csv("features.csv", float_precision="round_trip")
scores, warnings = model_a.score(inputs, load_fitted("A"))
```

B·C는 각각 `model_b`, `model_c`와 키 `"B"`, `"C"`를 사용합니다. 각 모델의 `fit(frame, y)`로 같은 구조를 다시 학습할 수 있습니다. 학습 표본을 바꾼다면 경고선도 새 검증 표본에서 다시 정해야 합니다. 저장된 학습 파일은 이 저장소에서 제공하는 버전을 사용합니다.
