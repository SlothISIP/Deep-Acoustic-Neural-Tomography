import React, { useState } from 'react';

const RoadmapCard = ({ phase, title, duration, status, children, isExpanded, onToggle }) => {
  const statusColors = {
    pending: 'bg-gray-100 border-gray-300',
    active: 'bg-blue-50 border-blue-400',
    complete: 'bg-green-50 border-green-400'
  };
  
  const statusBadge = {
    pending: 'bg-gray-200 text-gray-700',
    active: 'bg-blue-500 text-white',
    complete: 'bg-green-500 text-white'
  };

  return (
    <div className={`border-2 rounded-lg mb-4 ${statusColors[status]} transition-all duration-300`}>
      <div 
        className="p-4 cursor-pointer flex justify-between items-center"
        onClick={onToggle}
      >
        <div className="flex items-center gap-4">
          <span className="text-2xl font-bold text-gray-400">P{phase}</span>
          <div>
            <h3 className="font-bold text-lg">{title}</h3>
            <span className="text-sm text-gray-500">{duration}</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className={`px-3 py-1 rounded-full text-xs font-semibold ${statusBadge[status]}`}>
            {status.toUpperCase()}
          </span>
          <span className="text-xl">{isExpanded ? '▼' : '▶'}</span>
        </div>
      </div>
      {isExpanded && (
        <div className="px-4 pb-4 border-t border-gray-200 pt-4">
          {children}
        </div>
      )}
    </div>
  );
};

const Task = ({ number, title, description, libs, validation, critical }) => (
  <div className={`mb-4 p-3 rounded-lg ${critical ? 'bg-red-50 border-l-4 border-red-400' : 'bg-white border-l-4 border-blue-300'}`}>
    <div className="flex items-start gap-3">
      <span className="bg-gray-700 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0">
        {number}
      </span>
      <div className="flex-1">
        <h4 className="font-semibold text-gray-800">{title}</h4>
        <p className="text-sm text-gray-600 mt-1">{description}</p>
        {libs && (
          <div className="mt-2">
            <span className="text-xs font-semibold text-gray-500">Libraries: </span>
            <span className="text-xs text-blue-600">{libs}</span>
          </div>
        )}
        {validation && (
          <div className="mt-1 p-2 bg-yellow-50 rounded text-xs">
            <span className="font-semibold">✓ 검증: </span>{validation}
          </div>
        )}
      </div>
    </div>
  </div>
);

const MathBlock = ({ children }) => (
  <div className="bg-gray-800 text-green-400 p-3 rounded-lg font-mono text-sm my-2 overflow-x-auto">
    {children}
  </div>
);

const Milestone = ({ title, target }) => (
  <div className="bg-purple-100 border border-purple-300 rounded-lg p-3 mt-4">
    <div className="flex items-center gap-2">
      <span className="text-purple-600 text-xl">🎯</span>
      <div>
        <span className="font-bold text-purple-800">{title}</span>
        <span className="text-sm text-purple-600 ml-2">→ {target}</span>
      </div>
    </div>
  </div>
);

export default function AcousticTomographyRoadmap() {
  const [expanded, setExpanded] = useState({ 0: true, 1: true, 2: true, 3: true, 4: true });
  
  const togglePhase = (phase) => {
    setExpanded(prev => ({ ...prev, [phase]: !prev[phase] }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">
            🔊 Acoustic Neural Tomography
          </h1>
          <h2 className="text-xl text-blue-400">
            Physics-Rigorous Implementation Roadmap v3.1
          </h2>
          <p className="text-gray-400 mt-2 text-sm">
            "Simultaneous Reconstruction of Sound & Geometry via Structured Green's Learning"
          </p>
        </div>

        {/* Timeline Overview */}
        <div className="bg-slate-700 rounded-lg p-4 mb-6">
          <div className="flex justify-between items-center text-xs text-gray-300">
            <span>Month 1</span>
            <span>Month 4</span>
            <span>Month 7</span>
            <span>Month 11</span>
            <span>Month 13</span>
          </div>
          <div className="flex mt-2 h-3 rounded-full overflow-hidden">
            <div className="bg-blue-500 flex-1" title="Phase 1"></div>
            <div className="bg-green-500 flex-1" title="Phase 2"></div>
            <div className="bg-yellow-500 flex-1" title="Phase 3"></div>
            <div className="bg-red-500 w-1/6" title="Phase 4"></div>
          </div>
          <div className="flex mt-1 text-xs">
            <span className="flex-1 text-blue-400">BEM Engine</span>
            <span className="flex-1 text-green-400">Green-Net</span>
            <span className="flex-1 text-yellow-400">Neural Fields</span>
            <span className="w-1/6 text-red-400">Sim2Real</span>
          </div>
        </div>

        {/* Phase 0: Prerequisites */}
        <RoadmapCard 
          phase={0} 
          title="Prerequisites & Environment Setup" 
          duration="Week 0"
          status="active"
          isExpanded={expanded[0]}
          onToggle={() => togglePhase(0)}
        >
          <Task 
            number={1}
            title="Python 환경 구성"
            description="Python 3.9+, CUDA 지원 환경 확인. OpenCL 드라이버 필수."
            libs="conda, pip"
          />
          <Task 
            number={2}
            title="Core Dependencies 설치"
            description="BEM, 메쉬 생성, 딥러닝 프레임워크 설치"
            libs="bempp-cl, meshio, pygmsh, torch, numpy, scipy"
          />
          <Task 
            number={3}
            title="Optional: 시각화 & 실험 관리"
            description="결과 시각화 및 실험 추적"
            libs="matplotlib, plotly, wandb, tensorboard"
          />
          <MathBlock>
            pip install bempp-cl meshio pygmsh torch numpy scipy matplotlib
          </MathBlock>
        </RoadmapCard>

        {/* Phase 1 */}
        <RoadmapCard 
          phase={1} 
          title="BEM Physics Engine & Frequency Synthesis" 
          duration="Month 1-3"
          status="pending"
          isExpanded={expanded[1]}
          onToggle={() => togglePhase(1)}
        >
          <Task 
            number={1}
            title="Wedge Geometry BEM 검증"
            description="단순 무한 웨지(Infinite Wedge)에서 Helmholtz 방정식 풀이. 해석해(Macdonald)와 비교하여 BEM 정확도 검증."
            libs="bempp-cl, pygmsh"
            validation="해석해 대비 오차 < 5%"
            critical={true}
          />
          <MathBlock>
            ∇²p + k²p = 0, where k = 2πf/c
          </MathBlock>
          
          <Task 
            number={2}
            title="L-Shape Corridor 메쉬 생성"
            description="실제 실험 환경과 유사한 L-Shape 코너 메쉬 생성. Element size는 λ/6 이하로 설정."
            libs="pygmsh, meshio"
            validation="메쉬 품질 지표(Aspect Ratio) 확인"
          />
          
          <Task 
            number={3}
            title="Fresnel Number 기반 주파수 선정"
            description="F ≈ 1 조건에서 최적 주파수 대역 계산. 실용적 대역: 2-8 kHz."
            validation="Shadow Boundary에서 회절 신호 SNR > 10dB"
          />
          <MathBlock>
            F = a²/(λL) ≈ 1 → f_c = c·a²/L
          </MathBlock>
          
          <Task 
            number={4}
            title="Multi-Frequency BEM 솔버"
            description="2-8 kHz 대역 내 N개 주파수에서 Helmholtz 풀이. Adaptive sampling으로 계산량 최적화."
            libs="bempp-cl, joblib (병렬화)"
            validation="각 주파수에서 수렴 확인"
          />
          
          <Task 
            number={5}
            title="IDFT Time-Domain 합성"
            description="주파수 응답을 역푸리에 변환하여 RIR 생성. Causality(t<0에서 h(t)=0) 필수 확인."
            libs="numpy.fft, scipy.signal"
            validation="Causality 만족, 에너지 보존"
            critical={true}
          />
          <MathBlock>
            h(t) = IDFT{'{'}P(f₁), P(f₂), ..., P(fₙ){'}'} 
          </MathBlock>
          
          <Task 
            number={6}
            title="Dataset 생성 파이프라인"
            description="다양한 Source/Mic 위치 조합으로 10,000개 RIR 생성. HDF5 포맷 저장."
            libs="h5py, multiprocessing"
            validation="데이터 무결성, 재현성 확인"
          />
          
          <Milestone title="Phase 1 완료 기준" target="BEM RIR과 실측/해석해 일치" />
        </RoadmapCard>

        {/* Phase 2 */}
        <RoadmapCard 
          phase={2} 
          title="Structured Green's Function Learning" 
          duration="Month 4-6"
          status="pending"
          isExpanded={expanded[2]}
          onToggle={() => togglePhase(2)}
        >
          <Task 
            number={1}
            title="Image Source Method 구현"
            description="G_geometric (Direct + Reflection) 계산. 1차 반사까지 해석적으로 계산하여 고정."
            libs="numpy"
            validation="ISM vs BEM (LOS 영역) 오차 < 1%"
          />
          <MathBlock>
            G_total = G_geometric (Frozen) + G_diff (Learnable)
          </MathBlock>
          
          <Task 
            number={2}
            title="Diffraction MLP 설계"
            description="입력: (φ_inc, φ_obs, k), 출력: Diffraction Coefficient. UTD 해를 soft target으로 사전학습 고려."
            libs="torch.nn"
            validation="UTD 해와의 상관계수 > 0.9"
          />
          <MathBlock>
            Ĝ_diff = MLP(φ_inc, φ_obs, k) · exp(ikr)/r
          </MathBlock>
          
          <Task 
            number={3}
            title="Convolution Forward Model"
            description="입력 신호 s(t)와 G_total의 컨볼루션으로 측정 신호 y(t) 예측."
            libs="torch.nn.functional (conv1d)"
          />
          
          <Task 
            number={4}
            title="Green-Net 학습 루프"
            description="L2 Loss로 Diffraction MLP 학습. Learning rate scheduling 적용."
            libs="torch.optim (AdamW)"
            validation="Validation Loss 수렴"
          />
          <MathBlock>
            L = ||y(t) - s(t) * (G_geo + Ĝ_diff)||²
          </MathBlock>
          
          <Task 
            number={5}
            title="Ablation: Direct vs Structured"
            description="G_total 전체를 학습하는 Baseline과 비교. Structured 방식의 수렴 속도/정확도 우위 증명."
            validation="Structured > Direct (수렴 속도 2x↑)"
            critical={true}
          />
          
          <Milestone title="Phase 2 완료 기준" target="ICASSP 워크샵 페이퍼 Draft" />
        </RoadmapCard>

        {/* Phase 3 */}
        <RoadmapCard 
          phase={3} 
          title="Neural Fields with Implicit Geometry" 
          duration="Month 7-10"
          status="pending"
          isExpanded={expanded[3]}
          onToggle={() => togglePhase(3)}
        >
          <div className="bg-yellow-100 border border-yellow-400 rounded-lg p-3 mb-4">
            <span className="font-bold text-yellow-800">⚠️ CORE NOVELTY PHASE</span>
            <p className="text-sm text-yellow-700">이 Phase가 논문의 핵심 Contribution</p>
          </div>
          
          <Task 
            number={1}
            title="Fourier Feature Encoding"
            description="입력 좌표를 고주파 공간으로 매핑. σ ≈ k_max/(2π)로 설정."
            libs="torch"
            validation="고주파 신호 재현 가능 확인"
            critical={true}
          />
          <MathBlock>
            γ(x) = [cos(2πBx), sin(2πBx)], σ ≈ f_max/c
          </MathBlock>
          
          <Task 
            number={2}
            title="Joint Output Network 설계"
            description="입력: γ(x), t → 출력: (p, SDF). 두 출력이 공유하는 Feature Extractor + 분리된 Head."
            libs="torch.nn"
          />
          <MathBlock>
            f_θ: (γ(x), t) → (Pressure p, SDF s)
          </MathBlock>
          
          <Task 
            number={3}
            title="Eikonal Loss 구현"
            description="SDF의 gradient norm = 1 제약. torch.autograd.grad 사용."
            libs="torch.autograd"
            validation="|∇s| - 1 ≈ 0 (전체 영역)"
          />
          <MathBlock>
            L_geo = || |∇s(x)| - 1 ||²
          </MathBlock>
          
          <Task 
            number={4}
            title="Helmholtz PDE Loss 구현"
            description="2차 미분 계산하여 파동 방정식 만족 여부 검증."
            libs="torch.autograd (2nd derivative)"
            validation="PDE residual < 1e-3"
          />
          <MathBlock>
            L_Helmholtz = || ∇²p + k²p ||²
          </MathBlock>
          
          <Task 
            number={5}
            title="Boundary Condition Loss"
            description="SDF ≈ 0 영역에서 Neumann/Robin BC 적용. ∇s 방향이 법선."
            libs="torch.autograd"
            critical={true}
          />
          <MathBlock>
            L_BC = || ∂p/∂n + ikβp ||² at s(x)≈0
          </MathBlock>
          
          <Task 
            number={6}
            title="Multi-Loss Balancing"
            description="GradNorm 또는 Adaptive Weighting으로 Loss 균형 조절."
            libs="custom implementation"
            validation="모든 Loss 동시 수렴"
            critical={true}
          />
          <MathBlock>
            L_total = L_data + λ₁L_Helmholtz + λ₂L_geo + λ₃L_BC
          </MathBlock>
          
          <Task 
            number={7}
            title="Incremental Integration 학습"
            description="Step-by-step으로 Loss 추가. (1)Data만 → (2)+Eikonal → (3)+Helmholtz → (4)+BC"
            validation="각 단계에서 안정적 수렴"
          />
          
          <Milestone title="Phase 3 완료 기준" target="Simulation 데이터에서 SDF 복원 성공" />
        </RoadmapCard>

        {/* Phase 4 */}
        <RoadmapCard 
          phase={4} 
          title="Sim2Real & Cycle-Consistency Validation" 
          duration="Month 11-13"
          status="pending"
          isExpanded={expanded[4]}
          onToggle={() => togglePhase(4)}
        >
          <Task 
            number={1}
            title="실험 환경 구축"
            description="L-Shape 코너, Bluetooth 스피커, 스마트폰 마이크. Chirp 신호 (2-8 kHz)."
            validation="SNR > 20dB"
          />
          
          <Task 
            number={2}
            title="ARCore 기반 Pose 수집"
            description="스마트폰 궤적(Trajectory)과 오디오 동기화. Timestamp 정밀도 < 10ms."
            libs="Android ARCore API"
          />
          
          <Task 
            number={3}
            title="Pose Refinement"
            description="초기 10프레임 LOS 영역에서 Calibration 후, Shadow 영역 진입."
            libs="torch.optim"
            validation="위치 오차 < 5cm"
          />
          
          <Task 
            number={4}
            title="Inverse Pass: Real Audio → SDF"
            description="학습된 네트워크로 실제 소리에서 기하구조 추정."
          />
          
          <Task 
            number={5}
            title="Forward Pass: SDF → BEM → Simulated Audio"
            description="추정된 SDF를 BEM에 입력하여 가상 소리 생성."
            libs="bempp-cl"
          />
          
          <Task 
            number={6}
            title="Cycle-Consistency 검증"
            description="실제 소리와 가상 소리의 일치 여부 확인. 이것이 논문의 최종 증명."
            validation="Cycle Loss < threshold"
            critical={true}
          />
          <MathBlock>
            y_real ≈ y_sim = BEM(SDF_pred)
          </MathBlock>
          
          <Milestone title="Phase 4 완료 기준" target="CVPR 투고 / Nature Comms Draft" />
        </RoadmapCard>

        {/* Summary */}
        <div className="bg-slate-700 rounded-lg p-6 mt-6">
          <h3 className="text-xl font-bold text-white mb-4">📋 Key Deliverables</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="bg-slate-600 p-3 rounded">
              <span className="text-blue-400 font-semibold">Year 1 Q2:</span>
              <p className="text-gray-300">ICASSP Workshop Paper (방법론 검증)</p>
            </div>
            <div className="bg-slate-600 p-3 rounded">
              <span className="text-green-400 font-semibold">Year 2 Q2:</span>
              <p className="text-gray-300">CVPR Full Paper (Core Contribution)</p>
            </div>
            <div className="bg-slate-600 p-3 rounded">
              <span className="text-yellow-400 font-semibold">Year 3:</span>
              <p className="text-gray-300">Nature Communications (Application)</p>
            </div>
            <div className="bg-slate-600 p-3 rounded">
              <span className="text-purple-400 font-semibold">GitHub:</span>
              <p className="text-gray-300">Open-source Implementation</p>
            </div>
          </div>
        </div>

        {/* One-liner */}
        <div className="mt-6 p-4 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
          <p className="text-white text-center font-semibold">
            "We jointly reconstruct acoustic fields and scene geometry by learning only the diffraction residual atop analytical Green's functions, while enforcing Helmholtz PDE and Eikonal constraints."
          </p>
        </div>

        <p className="text-center text-gray-500 text-xs mt-6">
          Acoustic Neural Tomography Roadmap v3.1 | Target: CVPR Oral / Nature Communications
        </p>
      </div>
    </div>
  );
}
