# BeamDiff Introduction Draft

## Paper scope

BeamDiff v1 targets **BFA augmentation for BFI-based human activity sensing**.
It does not claim to generate complete IEEE 802.11 beamforming-feedback packets
for communication applications. In the final manuscript, replace the provisional
citation keys below with the keys used in the paper's BibTeX file.

## English draft

Wi-Fi sensing has emerged as a practical approach to recognizing human activities
without requiring users to carry dedicated sensors. Most existing systems rely on
channel state information (CSI), which captures fine-grained amplitude and phase
variations caused by human motion. Despite its sensing capability, CSI acquisition
is often tied to specific chipsets, modified firmware, or specialized collection
tools, which limits deployment on standard commercial Wi-Fi devices
\cite{sensefi}. This dependency also makes it difficult to collect sufficiently
large and diverse labeled datasets across users, environments, and device
configurations.

Beamforming feedback information (BFI) provides a more accessible sensing
representation. During standard IEEE 802.11ac/ax channel sounding, a beamformee
reports compressed beamforming information that can be captured without
firmware-level CSI extraction. In particular, the quantized beamforming feedback
angles (BFAs) encode the directional structure of the wireless channel and can be
directly organized as temporal sensing tensors. Wi-BFI provides an open-source
pipeline for extracting these angles and reconstructing the corresponding complex
beamforming matrix, while BeamSense demonstrates that BFAs can support human
activity recognition using standard-compliant Wi-Fi feedback
\cite{wibfi,beamsense}. However, BFI-based sensing still requires labeled
measurements for each activity and deployment condition. Its practical
accessibility therefore reduces the collection burden but does not eliminate the
need for effective data augmentation.

Generative augmentation offers a potential means of increasing RF sensing data
without repeating costly measurement campaigns. RF-Diffusion demonstrated that
diffusion models can synthesize time-series RF signals by modeling their temporal,
frequency-domain, and complex-valued properties \cite{rfdiffusion}. Motivated by
the accessibility of BFI, we first explored extending this approach to BFI by
using the reconstructed complex beamforming matrix as the generation target.
Although both CSI and the reconstructed matrix are complex-valued, they represent
different physical structures. CSI directly describes the channel response,
whereas the beamforming matrix represents channel directions and is subject to
normalization, geometric constraints, and phase ambiguity. In our preliminary
study, direct complex-matrix generation preserved tensor dimensions and basic
numerical validity but did not reliably preserve the temporal beam dynamics and
activity semantics required by downstream BFI sensing.

This observation motivates a representation-specific approach. The sensing model
does not consume packet headers or a reconstructed channel response; it directly
uses quantized BFAs. These angles are circular variables with channel-dependent
quantization ranges, and their frame-to-frame changes carry the temporal patterns
induced by human motion. Treating them as unconstrained real-valued features can
produce artificial discontinuities at quantization boundaries, while generating
each frame independently can destroy temporal consistency. A BFI sensing
generator should therefore model circular angle transitions, preserve the local
channel state from which a sequence begins, and retain the semantic information
associated with the requested activity.

To address these requirements, we present **BeamDiff**, a sensing-aware diffusion
framework designed for BFA sequence augmentation. BeamDiff retains the first real
BFA frame as an anchor and generates the nine circular frame-to-frame transitions
of a ten-frame sensing window. Its conditional denoiser incorporates the anchor,
the diffusion timestep, and the activity label. In addition to the standard
noise-prediction objective, BeamDiff employs clean-delta reconstruction and
temporal-fidelity losses to preserve BFA dynamics. A frozen BeamSense classifier
provides sensing supervision so that the generated sequence retains the intended
activity semantics. The generated transitions are accumulated from the anchor
and mapped back to the valid BFA quantization ranges, producing tensors that can
be directly used by existing BFI sensing models.

We evaluate BeamDiff on the HAR-1 and HAR-3 subsets of the CSI-BFI-HAR dataset
using fixed real training, validation, and test partitions. On HAR-1, adding one
synthetic BFA window per real training window improves the mean BeamSense accuracy
from 91.62% to 94.63% over five classifier initialization seeds, corresponding to
a mean paired gain of 3.00 percentage points; four of the five seeds improve. On
HAR-3, an initial paired evaluation improves accuracy from 98.70% to 99.65%,
reducing the number of test errors from 81 to 22. These results indicate that
BFA-specific, sensing-aware generation can improve in-domain activity recognition,
while our cross-participant measurements also reveal that generalization to
unseen subjects remains a separate and substantially harder problem.

The main contributions of this work are as follows:

1. We identify the structural mismatch between generic complex RF generation and
   BFI sensing augmentation, showing that numerical validity of a generated
   complex beamforming matrix does not by itself guarantee preservation of
   sensing-relevant beam dynamics.
2. We introduce BeamDiff, a BFA-specific conditional diffusion framework that
   models shortest-path circular angle transitions from a real first-frame anchor
   rather than independently generating quantized BFA frames.
3. We develop a sensing-aware training objective that combines diffusion noise
   prediction, clean-delta reconstruction, temporal-fidelity regularization, and
   frozen-classifier supervision to preserve both signal dynamics and activity
   semantics.
4. We conduct paired downstream evaluations in two BFI activity-sensing
   environments and analyze generated-data fidelity, seed-dependent behavior,
   and the remaining limitation under cross-participant evaluation.

## 한국어 의미본

Wi-Fi 센싱은 사용자가 별도의 센서를 착용하지 않고도 행동을 인식할 수 있는
기술로 주목받고 있다. 기존 시스템 대부분은 사람의 움직임에 따른 세밀한 진폭과
위상 변화를 제공하는 CSI를 사용한다. 그러나 CSI를 수집하려면 특정 칩셋, 수정된
펌웨어 또는 전용 수집 도구가 필요한 경우가 많아 상용 Wi-Fi 장치에서의 활용이
제한된다. 또한 사용자, 환경 및 장치 구성이 달라질 때마다 충분한 양의 라벨 데이터를
수집하는 데 상당한 비용이 필요하다.

BFI는 이보다 접근하기 쉬운 센싱 표현을 제공한다. IEEE 802.11ac/ax의 표준 채널
sounding 과정에서 단말은 압축된 beamforming 정보를 전송하며, 이를 펌웨어 수준의
CSI 추출 없이 수집할 수 있다. 특히 양자화된 BFA는 무선 채널의 방향 구조를
표현하며 시간 순서의 센싱 tensor로 직접 구성할 수 있다. Wi-BFI는 BFA 추출과
complex beamforming matrix 복원 기능을 제공하고, BeamSense는 BFA가 행동 인식에
사용될 수 있음을 보였다. 그러나 BFI의 수집 용이성만으로 사용자와 환경별 라벨
데이터 요구가 사라지는 것은 아니므로 효과적인 데이터 증강이 필요하다.

생성형 증강은 반복적인 RF 데이터 수집 없이 센싱 데이터를 확장할 수 있는 방법이다.
RF-Diffusion은 RF 신호의 시간, 주파수 및 복소수 특성을 모델링하여 time-series RF
신호를 생성할 수 있음을 보였다. 본 연구에서는 수집이 비교적 용이한 BFI까지 RF
생성 기술의 적용 범위를 확장하기 위해, 먼저 BFI로부터 복원한 complex beamforming
matrix를 RF-Diffusion의 생성 대상으로 적용하였다. 그러나 CSI와 복원된 matrix는
모두 복소수 형태이더라도 물리적 의미와 구조가 다르다. CSI는 채널 응답을 직접
표현하지만 beamforming matrix는 채널 방향을 나타내며 정규화, 기하학적 제약 및
phase ambiguity를 갖는다. 초기 실험에서 생성 matrix는 tensor shape와 기본적인
수치 조건을 만족했지만, BFI 센싱에 필요한 시간적 beam 변화와 activity 의미를
안정적으로 유지하지 못했다.

이러한 관찰은 BFI 표현에 특화된 생성 방법의 필요성을 보여준다. 본 연구에서
사용하는 센싱 모델은 packet header나 복원된 채널 응답이 아니라 양자화된 BFA를
직접 입력으로 사용한다. BFA는 channel마다 정해진 범위를 갖는 circular variable이며,
frame 간 변화에는 사람의 움직임으로 발생하는 시간 패턴이 포함된다. 이를 일반적인
실수 feature로 처리하면 양자화 경계에서 인위적인 불연속성이 발생할 수 있고, 각
frame을 독립적으로 생성하면 시간적 연속성이 손상될 수 있다. 따라서 BFI 센싱용
생성 모델은 circular angle transition, sequence가 시작되는 실제 채널 상태 및
activity semantic을 함께 보존해야 한다.

본 연구에서는 이러한 요구사항을 해결하기 위해 BFA sequence 증강에 특화된
sensing-aware diffusion framework인 **BeamDiff**를 제안한다. BeamDiff는 실제 첫
BFA frame을 anchor로 유지하고 10-frame window를 구성하는 이후 9개의 circular
frame-to-frame transition을 생성한다. Conditional denoiser는 anchor, diffusion
timestep 및 activity label을 조건으로 사용한다. 또한 기본 noise-prediction loss에
clean-delta reconstruction과 temporal-fidelity loss를 추가하여 BFA의 시간 변화를
보존한다. 고정된 BeamSense classifier의 sensing supervision을 이용해 합성 sequence가
요청한 activity 의미를 유지하도록 한다. 생성된 변화량은 anchor부터 누적되고 유효한
BFA 양자화 범위로 변환되므로 기존 BFI 센싱 모델에 직접 입력할 수 있다.

HAR-1과 HAR-3에서 고정된 real train, validation 및 test partition을 사용해 BeamDiff를
평가하였다. HAR-1에서 실제 train window와 동일한 수의 합성 window를 추가한 결과,
5개 classifier initialization seed의 평균 BeamSense 정확도가 91.62%에서 94.63%로
증가하여 평균 3.00%p의 paired gain을 보였고 5개 seed 중 4개에서 성능이 향상됐다.
HAR-3의 초기 paired 평가에서는 정확도가 98.70%에서 99.65%로 증가했고 test 오류가
81개에서 22개로 감소했다. 이 결과는 BFA에 특화된 sensing-aware generation이
in-domain 행동 인식 성능을 향상시킬 수 있음을 보여준다. 반면 cross-participant
결과는 새로운 사용자에 대한 일반화가 별도로 해결해야 할 더 어려운 문제임을 보여준다.

본 연구의 주요 기여는 다음과 같다.

1. 일반적인 complex RF generation과 BFI 센싱 증강 사이의 구조적 차이를 분석하고,
   생성된 complex beamforming matrix의 수치적 유효성만으로는 센싱에 필요한 beam
   dynamics 보존을 보장할 수 없음을 확인한다.
2. 양자화된 BFA frame을 독립적으로 생성하는 대신 실제 첫 frame anchor로부터
   shortest-path circular angle transition을 생성하는 BFA 특화 conditional diffusion
   framework인 BeamDiff를 제안한다.
3. Diffusion noise prediction, clean-delta reconstruction, temporal-fidelity
   regularization 및 frozen-classifier supervision을 결합하여 신호의 시간 변화와
   activity 의미를 함께 보존하는 sensing-aware objective를 설계한다.
4. 두 BFI 행동 센싱 데이터에서 paired downstream evaluation을 수행하고, 합성 데이터
   fidelity, seed에 따른 변화 및 cross-participant 조건의 한계를 분석한다.

## Citation mapping

- `sensefi`: SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing
- `wibfi`: Wi-BFI: Extracting the IEEE 802.11 Beamforming Feedback Information from Commercial Wi-Fi Devices
- `beamsense`: BeamSense: Rethinking Wireless Sensing with MU-MIMO Wi-Fi Beamforming Feedback
- `rfdiffusion`: RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion
