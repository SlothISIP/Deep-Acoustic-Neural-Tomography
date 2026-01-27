# Acoustic Neural Tomography: Implementation Roadmap v3.1

## "Simultaneous Reconstruction of Sound & Geometry via Structured Green's Learning"

**Target Publication:** CVPR (Oral) / Nature Communications  
**Timeline:** 13 Months  
**Core Contribution:** 소리만으로 보이지 않는 기하구조를 복원하는 물리 기반 딥러닝 프레임워크

---

## Executive Summary

이 로드맵은 "Deep Acoustic Diffraction Tomography" 연구를 Python으로 구현하기 위한 단계별 실행 계획이다. 핵심 아이디어는 단순하다: 소리의 회절(Diffraction) 패턴을 분석하여, 직접 볼 수 없는 장애물의 형태를 복원한다. 이를 위해 우리는 전통적인 블랙박스 딥러닝이 아닌, 물리 방정식(Helmholtz, Eikonal)을 네트워크 구조와 손실 함수에 내재시키는 Physics-Informed 접근법을 채택한다.

전체 구현은 크게 네 단계로 나뉜다. 첫째, BEM(Boundary Element Method) 기반의 물리 시뮬레이션 엔진을 구축한다. 둘째, 알려진 물리량은 고정하고 회절 성분만 학습하는 Structured Green's Function 네트워크를 개발한다. 셋째, 음압과 기하구조(SDF)를 동시에 출력하는 Neural Field를 구현하며, 여기에 Helmholtz PDE Loss와 Eikonal Loss를 결합한다. 넷째, 실제 실험 데이터로 Cycle-Consistency를 검증하여 논문의 최종 증거를 확보한다.

---

## Phase 0: Prerequisites & Environment Setup

**기간:** Week 0 (시작 전 준비)

구현을 시작하기 전에 적절한 개발 환경을 구축해야 한다. 이 연구의 핵심 도구는 bempp-cl이라는 Boundary Element Method 라이브러리다. 이 라이브러리는 OpenCL을 통해 GPU 가속을 지원하므로, 먼저 시스템에 OpenCL 드라이버가 올바르게 설치되어 있는지 확인해야 한다. NVIDIA GPU를 사용한다면 CUDA 드라이버와 함께 OpenCL 지원이 포함되어 있을 것이다.

Python 버전은 3.9 이상을 권장한다. 가상 환경을 생성한 후, 핵심 의존성 패키지들을 설치한다. bempp-cl은 BEM 시뮬레이션의 핵심이고, meshio와 pygmsh는 3D 메쉬 생성 및 처리를 담당한다. PyTorch는 딥러닝 모델 구현에 사용되며, numpy와 scipy는 수치 연산의 기반이 된다. 추가로 matplotlib과 plotly를 설치하면 결과 시각화에 유용하고, wandb나 tensorboard를 통해 실험을 체계적으로 관리할 수 있다.

환경 구축이 완료되면, 간단한 테스트로 bempp-cl이 정상 작동하는지 확인한다. 단순한 구(Sphere) 표면에서 Laplace 방정식을 풀어보고, 해석해와 비교하여 오차가 1% 이내인지 검증하는 것이 좋은 시작점이다.

---

## Phase 1: BEM Physics Engine & Frequency Synthesis

**기간:** Month 1-3  
**목표:** 물리적으로 정확한 Room Impulse Response(RIR) 데이터셋 생성

### 1.1 Wedge Geometry BEM 검증

모든 것의 시작은 검증이다. 복잡한 L-Shape 환경으로 바로 뛰어들기 전에, 해석해(Analytical Solution)가 존재하는 단순한 기하구조에서 BEM 구현의 정확성을 확인해야 한다. 무한 웨지(Infinite Wedge)는 이 목적에 완벽한 테스트 케이스다. Macdonald가 1915년에 유도한 웨지 회절 공식과 BEM 솔루션을 비교하여, 오차가 5% 이내임을 확인한다.

이 단계에서 주의할 점은 Sommerfeld Radiation Condition의 올바른 구현이다. 무한 영역에서 파동이 외부로 발산하는 조건을 만족시키지 않으면, BEM 솔루션이 비물리적인 결과를 낳는다. bempp-cl에서는 이를 위해 적절한 Green's Function을 선택해야 한다.

### 1.2 L-Shape Corridor 메쉬 생성

검증이 완료되면 실제 실험 환경을 모사하는 L-Shape 코너 메쉬를 생성한다. pygmsh를 사용하여 CAD 스타일로 기하구조를 정의하고, meshio를 통해 bempp-cl이 읽을 수 있는 포맷으로 변환한다.

메쉬 해상도는 시뮬레이션 정확도와 계산 시간 사이의 트레이드오프다. 경험적으로, 메쉬 요소 크기는 최소 파장의 1/6 이하여야 한다. 8kHz 소리의 파장은 약 4.3cm이므로, 요소 크기는 7mm 이하로 설정한다. 이보다 큰 요소를 사용하면 수치적 분산(Numerical Dispersion)이 발생하여 결과를 신뢰할 수 없게 된다.

### 1.3 Fresnel Number 기반 주파수 선정

회절 현상의 강도는 주파수에 따라 극적으로 변한다. Fresnel Number F = a²/(λL)가 약 1인 조건에서 회절이 가장 복잡한 양상을 보인다. 여기서 a는 장애물의 특성 길이, λ는 파장, L은 관측 거리다.

그러나 현실적인 실험에서는 이 조건만 고려할 수 없다. 너무 낮은 주파수는 공간 분해능이 떨어지고, 너무 높은 주파수는 대기 감쇠가 심해진다. 2kHz에서 8kHz 사이의 대역이 회절 분석과 실용성 모두를 만족시키는 최적의 영역이다. 이 대역 내에서 Chirp 신호를 설계하여 사용한다.

### 1.4 Multi-Frequency BEM 솔버

bempp-cl은 주파수 영역(Frequency Domain)에서 Helmholtz 방정식을 푼다. 시간 영역 RIR을 얻으려면 여러 주파수에서 솔루션을 구한 후 역푸리에 변환(IDFT)을 수행해야 한다.

원하는 RIR 길이가 T초라면, 주파수 해상도는 Δf = 1/T가 된다. 예를 들어 100ms의 RIR을 원한다면 10Hz 간격으로 주파수를 샘플링한다. 2kHz에서 8kHz 대역이라면 약 600개의 주파수에서 BEM을 풀어야 한다.

이 계산은 병렬화가 가능하다. 각 주파수는 독립적이므로, joblib이나 multiprocessing을 사용하여 여러 코어에서 동시에 계산할 수 있다. GPU가 여러 개라면 각 GPU에 다른 주파수를 할당하는 것도 좋은 전략이다.

### 1.5 IDFT Time-Domain 합성

주파수 영역 솔루션을 시간 영역으로 변환하는 과정에서 가장 중요한 검증 항목은 인과율(Causality)이다. 물리적으로, 소리가 발생하기 전(t < 0)에는 신호가 존재할 수 없다. IDFT 결과에서 t < 0 영역의 값이 수치적 오차 수준(< 1e-10)인지 반드시 확인해야 한다.

인과율이 깨지는 주요 원인은 위상(Phase) 처리 오류다. 복소수 주파수 응답의 위상이 올바르게 설정되지 않으면 시간 영역에서 비인과적 신호가 나타난다. 또한, 음의 주파수 성분은 양의 주파수의 복소 켤레(Complex Conjugate)로 채워야 실수 신호를 얻을 수 있다.

### 1.6 Dataset 생성 파이프라인

검증된 BEM-IDFT 파이프라인으로 대규모 데이터셋을 생성한다. 다양한 Source-Microphone 위치 조합에서 10,000개의 RIR을 생성하는 것이 목표다. 데이터는 HDF5 포맷으로 저장하여 효율적인 I/O를 보장한다.

각 데이터 샘플에는 RIR 자체뿐 아니라 메타데이터(Source 위치, Mic 위치, 주파수 대역, 메쉬 파라미터 등)도 함께 저장한다. 이는 나중에 실험 재현성을 보장하고, 다양한 조건에서의 성능 분석을 가능하게 한다.

**Phase 1 완료 기준:** BEM으로 생성한 RIR이 (1) 웨지 해석해와 5% 이내 일치, (2) 인과율 만족, (3) 에너지 보존 검증 통과.

---

## Phase 2: Structured Green's Function Learning

**기간:** Month 4-6  
**목표:** 물리적 구조를 반영한 회절 학습 네트워크 개발

### 2.1 Image Source Method 구현

Structured Green's Function의 핵심 아이디어는 "아는 것은 고정하고, 모르는 것만 학습한다"이다. 직접음(Direct Sound)과 1차 반사음(Specular Reflection)은 기하광학으로 정확히 계산할 수 있다. 이를 Image Source Method(ISM)로 구현하여 G_geometric으로 고정한다.

ISM은 가상의 음원(Image Source)을 반사면 너머에 배치하여 반사를 모델링한다. 1차 반사까지만 고려하면 계산이 단순하고, 대부분의 에너지를 포착할 수 있다. 이 G_geometric은 학습 과정에서 gradient update를 받지 않는다.

### 2.2 Diffraction MLP 설계

회절 성분 G_diff는 복잡한 각도 의존성을 가지므로 신경망으로 근사한다. 입력은 입사각(φ_inc), 관측각(φ_obs), 파수(k)의 세 변수다. 출력은 복소수 형태의 Diffraction Coefficient다.

네트워크 구조는 단순한 MLP로 충분하다. 4-5개의 hidden layer에 각 256개의 뉴런, ReLU 또는 SIREN activation을 사용한다. 중요한 것은 출력 형태다. Diffraction Coefficient는 exp(ikr)/r 형태의 발산 파동과 곱해지므로, 네트워크는 이 기본 형태 위에 "보정(Correction)"만 학습하면 된다.

선택적으로, UTD(Uniform Theory of Diffraction) 해석해를 soft target으로 사용하여 사전학습(Pre-training)할 수 있다. 이렇게 하면 학습 초기 수렴이 빨라지고, 물리적으로 타당한 솔루션 공간 내에서 탐색이 이루어진다.

### 2.3 Convolution Forward Model

학습된 G_total = G_geometric + G_diff를 사용하여 측정 신호를 예측한다. 입력 신호 s(t)와 G_total의 컨볼루션으로 y_hat(t)를 계산하고, 실제 측정 y(t)와의 L2 Loss를 최소화한다.

PyTorch에서는 torch.nn.functional.conv1d를 사용하여 효율적으로 구현할 수 있다. 주의할 점은 컨볼루션의 padding과 stride 설정이다. 'same' padding을 사용하여 출력 길이가 입력과 동일하게 유지되도록 한다.

### 2.4 Green-Net 학습 루프

AdamW 옵티마이저를 사용하여 학습한다. Learning rate는 1e-4에서 시작하여 CosineAnnealing 스케줄러로 점진적으로 감소시킨다. Batch size는 GPU 메모리가 허용하는 한 크게 설정하는 것이 좋다.

학습 중 validation loss를 모니터링하여 overfitting을 감지한다. Early stopping을 적용하되, patience를 충분히 (50 epoch 이상) 설정하여 조기 종료로 인한 underfitting을 방지한다.

### 2.5 Ablation Study: Direct vs Structured

Structured 접근법의 효과를 입증하기 위해, G_total 전체를 처음부터 학습하는 Baseline 모델과 비교한다. 동일한 데이터와 학습 조건에서 (1) 수렴 속도, (2) 최종 정확도, (3) 일반화 성능을 측정한다.

예상 결과: Structured 방식이 수렴 속도 2배 이상 빠르고, 특히 학습 데이터에 없는 새로운 기하구조에서 일반화 성능이 월등히 우수할 것이다. 이 결과는 논문의 핵심 ablation study가 된다.

**Phase 2 완료 기준:** Green-Net이 (1) Validation Loss 수렴, (2) UTD 해와 높은 상관관계(r > 0.9), (3) Baseline 대비 우수한 성능. 이 시점에서 ICASSP 워크샵 페이퍼 초안 작성.

---

## Phase 3: Neural Fields with Implicit Geometry

**기간:** Month 7-10  
**목표:** 소리와 기하구조를 동시에 복원하는 Physics-Informed Neural Field 구현

### ⚠️ 이 Phase가 논문의 핵심 Contribution이다

### 3.1 Fourier Feature Encoding

일반적인 MLP는 고주파 함수를 학습하는 데 어려움을 겪는다. 이를 Spectral Bias라 한다. 파동 현상은 본질적으로 고주파이므로(e^{ikx} 형태), 이 문제를 해결하지 않으면 PINN이 제대로 작동하지 않는다.

Fourier Feature Mapping은 입력 좌표를 고차원 주파수 공간으로 변환하여 이 문제를 해결한다. 좌표 x를 γ(x) = [cos(2πBx), sin(2πBx)]로 매핑한다. 여기서 B는 주파수 행렬로, Gaussian 분포에서 샘플링된다.

핵심 하이퍼파라미터는 B의 스케일 σ다. 이 값이 너무 작으면 저주파만, 너무 크면 고주파만 학습한다. 물리적 직관을 활용하면 최적값을 바로 도출할 수 있다: σ ≈ k_max/(2π) = f_max/c. 8kHz 대역이라면 σ ≈ 23 m⁻¹이다. 이렇게 물리에서 직접 하이퍼파라미터를 유도하는 것은 논문에서 강조할 만한 포인트다.

### 3.2 Joint Output Architecture

이전 버전(v3.0)에서는 d_edge(장애물까지의 거리)를 입력으로 사용했다. 이는 논리적 오류다. 역문제(Inverse Problem)에서 우리가 찾고자 하는 것이 바로 장애물의 위치이기 때문이다.

v3.1에서는 이를 수정하여, 네트워크가 음압 p와 함께 기하구조를 나타내는 SDF(Signed Distance Function)를 동시에 출력하도록 설계한다. SDF는 각 점에서 가장 가까운 표면까지의 부호 있는 거리로, 표면에서 0, 내부에서 음수, 외부에서 양수 값을 갖는다.

네트워크 구조는 공유 Feature Extractor와 두 개의 분리된 Head로 구성된다. Feature Extractor는 Fourier Features를 받아 고수준 표현을 학습하고, Acoustic Head는 음압 p를, Geometry Head는 SDF s를 출력한다.

### 3.3 Eikonal Loss 구현

SDF가 물리적으로 유효하려면 그 gradient의 크기가 모든 점에서 1이어야 한다. 이를 Eikonal 방정식이라 하며, L_geo = ‖|∇s| - 1‖²로 손실 함수에 반영한다.

PyTorch에서는 torch.autograd.grad를 사용하여 네트워크 출력의 gradient를 계산할 수 있다. create_graph=True 옵션을 설정하여 gradient의 gradient도 계산 가능하게 해야 한다.

### 3.4 Helmholtz PDE Loss 구현

PINN의 핵심은 물리 방정식 자체를 손실 함수로 사용하는 것이다. Helmholtz 방정식 ∇²p + k²p = 0이 만족되지 않으면 페널티를 부과한다: L_Helmholtz = ‖∇²p + k²p‖².

2차 미분 계산은 torch.autograd.grad를 두 번 호출하여 구현한다. 첫 번째 호출로 1차 미분을 구하고, 그 결과에 다시 grad를 적용하여 2차 미분을 얻는다. Laplacian은 각 좌표에 대한 2차 미분의 합이다.

### 3.5 Boundary Condition Loss

소리와 기하구조를 연결하는 핵심 고리는 경계 조건(Boundary Condition)이다. SDF가 0에 가까운 영역, 즉 물체 표면에서 음향 경계 조건이 만족되어야 한다.

Rigid wall의 경우 Neumann 조건 ∂p/∂n = 0을 적용한다. 여기서 법선 방향 n은 SDF의 gradient로부터 직접 얻을 수 있다: n = ∇s/|∇s|. 흡음재가 있는 경우 Robin 조건 ∂p/∂n + ikβp = 0을 사용하며, β는 정규화된 어드미턴스다.

L_BC는 s(x) ≈ 0인 점들에서만 계산한다. 이를 위해 soft mask를 사용하거나, SDF 값이 임계값 이하인 점들을 샘플링한다.

### 3.6 Multi-Loss Balancing

총 손실 함수는 L_total = L_data + λ₁L_Helmholtz + λ₂L_geo + λ₃L_BC다. 네 개의 손실 항이 서로 다른 스케일과 수렴 속도를 가지므로, 가중치 λ를 잘 조절해야 한다.

가장 단순한 방법은 고정된 λ 값을 사용하는 것이지만, 이는 최적이 아니다. 학습 초기에는 L_data가 지배적이고, 물리 손실들은 무시당한다. Adaptive Weighting 기법을 사용하면 각 손실의 크기에 따라 가중치를 동적으로 조절할 수 있다.

GradNorm 알고리즘은 각 손실 항의 gradient norm을 비슷하게 맞추도록 λ를 학습한다. 더 간단한 방법으로는 λ_i(t) = λ_i⁰ · (L_i(0)/L_i(t))^α 형태의 스케줄링이 있다. 손실이 줄어들수록 가중치를 높여 균형을 맞춘다.

### 3.7 Incremental Integration 학습

모든 손실을 처음부터 동시에 사용하면 학습이 불안정해질 수 있다. 점진적 통합(Incremental Integration) 전략을 권장한다.

Step 1: L_data만 사용하여 기본적인 fitting 수행.  
Step 2: L_geo(Eikonal)를 추가하여 SDF가 유효한 형태를 갖도록 유도.  
Step 3: L_Helmholtz를 추가하여 물리 방정식 만족.  
Step 4: L_BC를 추가하여 경계에서 소리-기하 연결.

각 단계에서 충분히 수렴한 후 다음 손실을 추가한다. 이렇게 하면 디버깅도 용이하고, 각 손실 항의 효과를 명확히 분석할 수 있다.

**Phase 3 완료 기준:** (1) Simulation 데이터에서 SDF 복원 IoU > 0.8, (2) Helmholtz residual < 1e-3, (3) 모든 손실 동시 수렴. 이 시점에서 CVPR 논문의 핵심 실험 완료.

---

## Phase 4: Sim2Real & Cycle-Consistency Validation

**기간:** Month 11-13  
**목표:** 실제 실험 데이터로 방법론 검증, 논문 최종 증거 확보

### 4.1 실험 환경 구축

실험실에 L-Shape 코너를 구성한다. 벽면 재질은 가급적 단단한 것(합판, MDF)을 사용하여 Rigid wall 가정에 부합하게 한다. 바닥에 30cm 격자를 그려 측정 위치를 정확히 제어한다.

음원으로는 Bluetooth 스피커를 사용하고, 수신기로는 스마트폰의 마이크를 활용한다. 스피커는 가급적 무지향성(Omnidirectional)에 가까운 것을 선택한다. Chirp 신호(2-8 kHz, 100ms)를 재생하고 녹음한다.

가장 중요한 것은 SNR이다. 조용한 환경에서 측정하고, 여러 번 반복 측정 후 평균을 취하여 노이즈를 줄인다. 목표 SNR은 20dB 이상이다.

### 4.2 ARCore 기반 Pose 수집

스마트폰의 위치를 추적하기 위해 ARCore API를 활용한다. ARCore는 Visual-Inertial Odometry를 사용하여 6-DoF pose를 실시간으로 추정한다. 오디오 녹음과 pose 데이터의 timestamp를 동기화하여 저장한다.

동기화 정밀도는 10ms 이내를 목표로 한다. 소리의 속도가 약 343m/s이므로, 10ms 오차는 약 3.4cm의 위치 오차에 해당한다. 이 정도면 우리의 분석에 충분하다.

### 4.3 Pose Refinement

ARCore의 pose 추정에는 드리프트(Drift)와 점핑(Jumping) 오차가 있다. 이를 보정하기 위해 초기 10프레임은 Line-of-Sight(직접음이 도달하는) 영역에서 측정한다. 이 영역에서는 직접음의 도착 시간으로부터 거리를 정확히 계산할 수 있으므로, ARCore pose를 calibration할 수 있다.

Calibration 후 Shadow 영역(직접음이 차단되는 영역)으로 진입한다. 이 영역에서 측정된 데이터가 실제 회절 분석에 사용된다.

### 4.4 Inverse Pass: Real Audio → SDF

학습된 네트워크에 실제 오디오 데이터를 입력하여 기하구조(SDF)를 추정한다. 이 과정은 추론(Inference)이므로 gradient 계산이 필요 없고, 빠르게 수행된다.

추정된 SDF로부터 iso-surface(s = 0인 면)를 추출하여 3D 메쉬로 시각화한다. Marching Cubes 알고리즘을 사용하면 SDF에서 직접 삼각 메쉬를 생성할 수 있다.

### 4.5 Forward Pass: SDF → BEM → Simulated Audio

추정된 기하구조가 물리적으로 타당한지 검증하기 위해, 이를 다시 BEM 시뮬레이터에 입력한다. Phase 1에서 구축한 파이프라인을 사용하여 "만약 이 기하구조가 맞다면 어떤 소리가 들릴까?"를 계산한다.

이 과정에서 추정된 SDF를 BEM 메쉬로 변환해야 한다. iso-surface 추출 후 메쉬 품질을 개선(smoothing, remeshing)하여 BEM에 적합한 형태로 만든다.

### 4.6 Cycle-Consistency 검증

최종 검증은 Cycle-Consistency다. 실제 측정 오디오 y_real과 시뮬레이션 오디오 y_sim이 일치하는지 확인한다. 완벽한 일치는 불가능하지만, correlation coefficient가 0.8 이상이면 성공으로 간주한다.

이 검증이 중요한 이유는 Ground Truth가 없기 때문이다. 실제 실험에서 "진짜" 기하구조는 알 수 없다(측정 오차 존재). 그러나 추정된 기하구조가 관측된 소리를 설명할 수 있다면, 그것은 물리적으로 타당한 솔루션이다. 이것이 역문제 검증의 표준 방법론이다.

**Phase 4 완료 기준:** (1) Real 데이터에서 SDF 복원, (2) Cycle-Consistency 검증 통과, (3) CVPR 투고용 논문 완성.

---

## Key Deliverables & Timeline

| Milestone | 시점 | 산출물 |
|-----------|------|--------|
| BEM 검증 완료 | Month 3 | 검증된 시뮬레이션 파이프라인, 10K RIR 데이터셋 |
| ICASSP 투고 | Month 6 | 워크샵 페이퍼 (Green-Net 방법론) |
| Neural Field 완성 | Month 10 | PINN 기반 Joint Learning 코드 |
| CVPR 투고 | Month 13 | Full Paper (전체 프레임워크) |
| Nature Comms | Year 3 | 응용 확장 (의료 초음파 등) |

---

## One-Line Contribution

> "We propose the first physics-rigorous framework that jointly reconstructs acoustic fields and scene geometry from monaural audio by learning only the diffraction residual atop analytical Green's functions, while enforcing Helmholtz PDE and Eikonal constraints."

이 한 문장이 Abstract의 첫 줄이자, 리뷰어가 "Accept"를 결정하는 핵심 근거다.

---

## Potential Risks & Mitigations

**Risk 1: BEM 수치 불안정성**  
Helmholtz 방정식은 특정 주파수(공진 주파수)에서 ill-conditioned 행렬을 생성할 수 있다. GMRES 솔버의 수렴 실패로 나타난다.  
*Mitigation:* Burton-Miller 공식 또는 CHIEF 방법으로 유일해 보장.

**Risk 2: PINN 수렴 실패**  
여러 손실 항의 충돌로 학습이 진동하거나 발산할 수 있다.  
*Mitigation:* Incremental Integration, Adaptive Weighting, Learning rate warmup.

**Risk 3: Sim2Real Gap**  
시뮬레이션과 현실의 차이(벽 재질, 온도, 습도 등)로 Transfer 실패.  
*Mitigation:* Domain Randomization (시뮬레이션에 노이즈 추가), Fine-tuning on real data.

**Risk 4: 계산 시간 초과**  
600개 주파수 × 10,000 샘플 = 6M BEM 솔브. 단일 GPU로 수개월 소요 가능.  
*Mitigation:* Adaptive frequency sampling, 클러스터 사용, Transfer learning으로 샘플 수 감소.

---

## Final Remarks

이 로드맵은 "소리로 보이지 않는 세상을 본다"는 비전을 13개월 안에 구현하기 위한 청사진이다. 각 단계는 명확한 검증 기준을 가지고 있으며, Critical Path를 따라 진행하면 논문 수준의 결과를 얻을 수 있다.

가장 중요한 것은 **Phase 1의 검증**이다. BEM이 정확하지 않으면 그 위에 쌓는 모든 것이 무너진다. Wedge에서의 해석해 비교를 반드시 통과한 후 다음으로 진행하라.

두 번째로 중요한 것은 **Phase 3의 Loss Balancing**이다. 여러 물리 손실을 동시에 최적화하는 것은 예술에 가깝다. Incremental Integration과 충분한 실험을 통해 안정적인 학습 레시피를 찾아야 한다.

마지막으로, 연구는 계획대로 되지 않는다. 예상치 못한 문제가 반드시 발생한다. 이 로드맵을 기반으로 하되, 유연하게 적응하라. 문제가 생기면 그것 자체가 새로운 연구 질문이 될 수 있다.

건투를 빈다.

---

*Acoustic Neural Tomography Implementation Roadmap v3.1*  
*Target: CVPR Oral / Nature Communications*  
*Last Updated: January 2026*
