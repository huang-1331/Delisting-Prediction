# 순손실 기반 12개월 완전자본잠식 예측

예측 기준일까지 공개된 최신 별도 재무제표로 위험 점수를 계산한다.

\[
\text{위험 점수} = -\frac{\text{최신 자기자본} + \text{최근 12개월 순이익}}{\text{최신 자산}}
\]

최근 12개월 순이익을 산출할 수 없으면 음의 자기자본/자산 비율을 사용한다. 점수가 경고선 이상이면 경고한다. 비교에 사용한 경고선은 학습셋 재현율 80% 기준에서 확정한 `-0.4501444339711528`이다.

| 평가 행 | 실제 부실 | 적중 | 오경보 | 누락 | 재현율 | 정밀도 |
|---:|---:|---:|---:|---:|---:|---:|
| 1,377 | 74 | 61 | 353 | 13 | 82.43% | 14.73% |

[모델 검증](../모델%20검증/)의 MLP와 동일한 기업·분기 테스트 행에 적용했다. 테스트 기업은 학습과 경고선 결정에서 제외했다. 기존 평가 표본을 재사용한 결과다.

## 실행

저장소 루트에서 프로젝트의 기존 스냅샷, 재무이력, 공개일 레지스트리를 전달한다. `refine.py`는 순손실 규칙을 포함한 원래 비교 실험을 실행한다. 순손실 행은 `annual_loss_projection`과 `floor=0.8`로 확인한다.

```bash
python -m pip install -r "모델 검증/requirements.txt"
python "순손실 알고리즘/refine.py" fit train_snapshots.csv.gz source_financial_history.csv.gz public_known_registry.csv loss_results
python "순손실 알고리즘/refine.py" test test_snapshots.csv.gz source_financial_history.csv.gz public_known_registry.csv loss_results
```

고정 경고선과 해당 점수의 테스트 집계는 `results/`에 있다.
