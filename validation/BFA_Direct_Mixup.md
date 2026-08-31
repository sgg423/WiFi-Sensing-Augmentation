# BFA 직접 증강 — 2026-08-31

RF-Diffusion/V 생성 없이 원본 quantized BFA를 같은 클래스끼리 혼합하는 대조 실험.
기존 circular-mixup 코드의 psi 주기 처리를 수정했다. 과거 생성 파일은 재사용하지 않는다.

- 입력 및 출력: `[N,10,234,4]`, uint16, phi/phi/psi/psi 순서.
- phi: 512-bin 원형 공간에서 가중 혼합. psi: 0–127 범위에서 선형 보간.
- 샘플별 혼합 계수 0.05–0.15, 시간/서브캐리어 전체에 같은 계수 적용.
- `--train-split-seed 111`: BeamSense random-window train fold만 peer로 사용.
- 검증·테스트 행은 그대로 보관되지만 학습 증강에 사용하지 않음.
- 클래스별 train peer 선택, singleton은 자기 자신 유지. 레이블과 source/window metadata 보존.
- `--seed 111`: 혼합 peer/계수 선택; classifier seed, augmentation subsampling seed와 별개.
- 첫 비교: 원본 train 28,529 vs 원본 28,529 + 혼합 2,852. 검증 6,064 / 테스트 6,082.
- 혼합 강도 5–15%와 추가 데이터 비율 10%는 다른 파라미터다.
- classifier model/split/augmentation seed 111; epochs 100, balanced class weight, normalization none.
- 기준 참고값: 고정 분할 baseline seed111 Accuracy 94.0973%. 동일 코드 실행 조건 확인 필요.

스크립트: `scripts/augment_bfa_circular_mixup.py`.
기존 파일을 덮어쓰지 않으며 split seed를 필수로 요구한다.
trainer는 직접 증강 파일의 split seed와 eligibility를 확인한다.

검증: `tests/test_bfa_direct_mixup.py` 2개 테스트 통과. 경계각, psi 비순환,
시간별 동일 혼합, train-only peer, metadata 보존, 반복 재현, 덮어쓰기 거부 확인.
실제 GPU 학습은 아직 실행하지 않았다.

이는 전체 시간 패턴을 보존한다고 보장하는 방법은 아니다. 서로 다른 행동 진행
시점의 같은 클래스 윈도우가 섞일 수 있으므로 최종 real test 성능으로 평가한다.
여러 방법 선택에는 validation을 사용하고, 반복 조회한 test 결과는 탐색적 결과로 다룬다.
