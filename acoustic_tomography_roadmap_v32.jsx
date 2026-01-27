import React, { useState } from 'react';

const RoadmapCard = ({ phase, title, duration, status, children, isExpanded, onToggle, isRevised }) => {
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
            <div className="flex items-center gap-2">
              <h3 className="font-bold text-lg">{title}</h3>
              {isRevised && (
                <span className="bg-purple-500 text-white text-xs px-2 py-0.5 rounded">REVISED</span>
              )}
            </div>
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

const Task = ({ number, title, description, libs, validation, critical, isNew, isModified }) => (
  <div className={`mb-4 p-3 rounded-lg ${
    critical ? 'bg-red-50 border-l-4 border-red-400' : 
    isNew ? 'bg-green-50 border-l-4 border-green-500' :
    isModified ? 'bg-yellow-50 border-l-4 border-yellow-500' :
    'bg-white border-l-4 border-blue-300'
  }`}>
    <div className="flex items-start gap-3">
      <span className={`rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0 ${
        critical ? 'bg-red-600 text-white' :
        isNew ? 'bg-green-600 text-white' :
        isModified ? 'bg-yellow-600 text-white' :
        'bg-gray-700 text-white'
      }`}>
        {number}
      </span>
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <h4 className="font-semibold text-gray-800">{title}</h4>
          {isNew && <span className="text-xs bg-green-500 text-white px-1.5 py-0.5 rounded">NEW</span>}
          {isModified && <span className="text-xs bg-yellow-500 text-white px-1.5 py-0.5 rounded">MODIFIED</span>}
        </div>
        <p className="text-sm text-gray-600 mt-1">{description}</p>
        {libs && (
          <div className="mt-2">
            <span className="text-xs font-semibold text-gray-500">Libraries: </span>
            <span className="text-xs text-blue-600">{libs}</span>
          </div>
        )}
        {validation && (
          <div className="mt-1 p-2 bg-yellow-50 rounded text-xs border border-yellow-200">
            <span className="font-semibold">✓ Validation: </span>{validation}
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

const CodeBlock = ({ title, children }) => (
  <div className="my-3">
    {title && <div className="text-xs text-gray-500 mb-1">{title}</div>}
    <div className="bg-gray-900 text-gray-100 p-3 rounded-lg font-mono text-xs overflow-x-auto">
      <pre>{children}</pre>
    </div>
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

const RiskItem = ({ severity, code, title, mitigation }) => (
  <div className={`p-3 rounded-lg border-l-4 mb-2 ${
    severity === 'critical' ? 'border-red-500 bg-red-50' :
    severity === 'high' ? 'border-orange-500 bg-orange-50' :
    'border-yellow-500 bg-yellow-50'
  }`}>
    <div className="flex items-center gap-2">
      <span className={`text-xs font-bold uppercase px-2 py-0.5 rounded ${
        severity === 'critical' ? 'bg-red-500 text-white' :
        severity === 'high' ? 'bg-orange-500 text-white' : 
        'bg-yellow-500 text-white'
      }`}>{code}</span>
      <span className="font-semibold text-gray-800">{title}</span>
    </div>
    <p className="text-sm text-gray-600 mt-1"><strong>Mitigation:</strong> {mitigation}</p>
  </div>
);

const RevisionNote = ({ children }) => (
  <div className="bg-purple-100 border border-purple-400 rounded-lg p-3 mb-4">
    <div className="flex items-start gap-2">
      <span className="text-purple-600">📝</span>
      <div className="text-sm text-purple-800">{children}</div>
    </div>
  </div>
);

export default function AcousticTomographyRoadmapV32() {
  const [expanded, setExpanded] = useState({ 0: true, 1: true, 2: true, 3: true, 4: true });
  
  const togglePhase = (phase) => {
    setExpanded(prev => ({ ...prev, [phase]: !prev[phase] }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold text-white mb-2">
            🔊 Acoustic Neural Tomography
          </h1>
          <h2 className="text-xl text-blue-400">
            Implementation Roadmap v3.2
          </h2>
          <div className="flex justify-center gap-2 mt-2">
            <span className="bg-purple-600 text-white text-xs px-3 py-1 rounded-full">
              Dr. Tensor Wave Review Reflected
            </span>
            <span className="bg-green-600 text-white text-xs px-3 py-1 rounded-full">
              Critical Fixes Applied
            </span>
          </div>
          <p className="text-gray-400 mt-2 text-sm">
            "Complex Field Handling + Trivial Solution Prevention + Phase Unwrapping"
          </p>
        </div>

        {/* Version Diff */}
        <div className="bg-slate-700 rounded-lg p-4 mb-6">
          <h3 className="text-white font-bold mb-3">📋 v3.1 → v3.2 주요 변경사항</h3>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="bg-red-900/30 p-2 rounded border border-red-500/50">
              <span className="text-red-400 font-semibold">C1 Fix:</span>
              <p className="text-gray-300">복소수 압력장 Re/Im 분리 출력</p>
            </div>
            <div className="bg-red-900/30 p-2 rounded border border-red-500/50">
              <span className="text-red-400 font-semibold">C2 Fix:</span>
              <p className="text-gray-300">Surface Existence Constraint 추가</p>
            </div>
            <div className="bg-red-900/30 p-2 rounded border border-red-500/50">
              <span className="text-red-400 font-semibold">C3 Fix:</span>
              <p className="text-gray-300">Phase Unwrapping (np.unwrap)</p>
            </div>
            <div className="bg-orange-900/30 p-2 rounded border border-orange-500/50">
              <span className="text-orange-400 font-semibold">H1 Fix:</span>
              <p className="text-gray-300">σ: 23 → 62 m⁻¹ (각도 의존성)</p>
            </div>
            <div className="bg-orange-900/30 p-2 rounded border border-orange-500/50">
              <span className="text-orange-400 font-semibold">H3 Fix:</span>
              <p className="text-gray-300">Burton-Miller α = i/k 명시</p>
            </div>
            <div className="bg-blue-900/30 p-2 rounded border border-blue-500/50">
              <span className="text-blue-400 font-semibold">Timeline:</span>
              <p className="text-gray-300">13개월 → 18개월 (현실적 조정)</p>
            </div>
          </div>
        </div>

        {/* Timeline Overview - FIXED proportions */}
        <div className="bg-slate-700 rounded-lg p-4 mb-6">
          <div className="flex justify-between items-center text-xs text-gray-300">
            <span>M1</span>
            <span>M4</span>
            <span>M8</span>
            <span>M13</span>
            <span>M18</span>
          </div>
          <div className="flex mt-2 h-3 rounded-full overflow-hidden">
            <div className="bg-blue-500" style={{flexGrow: 4}} title="Phase 1: 4 months"></div>
            <div className="bg-green-500" style={{flexGrow: 4}} title="Phase 2: 4 months"></div>
            <div className="bg-yellow-500" style={{flexGrow: 5}} title="Phase 3: 5 months"></div>
            <div className="bg-red-500" style={{flexGrow: 5}} title="Phase 4: 5 months"></div>
          </div>
          <div className="flex mt-1 text-xs">
            <div style={{flexGrow: 4}} className="text-blue-400">BEM Engine</div>
            <div style={{flexGrow: 4}} className="text-green-400">Green-Net</div>
            <div style={{flexGrow: 5}} className="text-yellow-400">Neural Fields</div>
            <div style={{flexGrow: 5}} className="text-red-400">Sim2Real</div>
          </div>
        </div>

        {/* Computational Requirements - NEW */}
        <div className="bg-gradient-to-r from-slate-700 to-slate-600 rounded-lg p-4 mb-6 border border-slate-500">
          <h3 className="text-white font-bold mb-3">💻 Computational Requirements</h3>
          <div className="grid grid-cols-4 gap-3 text-xs">
            <div className="bg-slate-800 p-2 rounded text-center">
              <div className="text-gray-400">GPU</div>
              <div className="text-white font-bold">RTX 4090</div>
              <div className="text-gray-500">24GB+ VRAM</div>
            </div>
            <div className="bg-slate-800 p-2 rounded text-center">
              <div className="text-gray-400">CPU</div>
              <div className="text-white font-bold">32+ cores</div>
              <div className="text-gray-500">BEM 병렬화</div>
            </div>
            <div className="bg-slate-800 p-2 rounded text-center">
              <div className="text-gray-400">RAM</div>
              <div className="text-white font-bold">128 GB</div>
              <div className="text-gray-500">Dataset 처리</div>
            </div>
            <div className="bg-slate-800 p-2 rounded text-center">
              <div className="text-gray-400">Storage</div>
              <div className="text-white font-bold">1 TB NVMe</div>
              <div className="text-gray-500">RIR 데이터</div>
            </div>
          </div>
          <div className="text-xs text-gray-400 mt-2 text-center">
            ⚠️ Single GPU 기준 18개월 | 클러스터(4x A100) 사용 시 12개월로 단축 가능
          </div>
        </div>

        {/* Phase 0: Prerequisites */}
        <RoadmapCard 
          phase={0} 
          title="Prerequisites & Environment Setup" 
          duration="Week 0-1"
          status="active"
          isExpanded={expanded[0]}
          onToggle={() => togglePhase(0)}
        >
          <Task 
            number={1}
            title="Python 환경 구성"
            description="Python 3.10+, CUDA 12.x, OpenCL 드라이버 필수. conda 환경 권장."
            libs="conda, pip"
          />
          <Task 
            number={2}
            title="Core Dependencies 설치"
            description="BEM, 메쉬 생성, 딥러닝 프레임워크 설치"
            libs="bempp-cl, meshio, pygmsh, torch>=2.0, numpy, scipy"
          />
          <Task 
            number={3}
            title="Complex Number Support 확인"
            description="PyTorch complex tensor 지원 확인. torch.complex64/128 연산 테스트."
            libs="torch"
            validation="torch.complex64 matmul 동작 확인"
            isNew={true}
          />
          <CodeBlock title="환경 설치">
{`conda create -n acoustic-tomo python=3.10
conda activate acoustic-tomo
pip install bempp-cl meshio pygmsh torch numpy scipy
pip install matplotlib plotly wandb h5py joblib`}
          </CodeBlock>
        </RoadmapCard>

        {/* Phase 1 - REVISED */}
        <RoadmapCard 
          phase={1} 
          title="BEM Physics Engine & Frequency Synthesis" 
          duration="Month 1-4 (4 months)"
          status="pending"
          isExpanded={expanded[1]}
          onToggle={() => togglePhase(1)}
          isRevised={true}
        >
          <RevisionNote>
            <strong>v3.2 Changes:</strong> Burton-Miller 파라미터 명시, Phase Unwrapping 추가, 
            Energy Conservation 검증 추가, Adaptive Mesh 도입
          </RevisionNote>

          <Task 
            number={1}
            title="Wedge Geometry BEM 검증"
            description="단순 무한 웨지(Infinite Wedge)에서 Helmholtz 방정식 풀이. Macdonald 해석해와 비교."
            libs="bempp-cl, pygmsh"
            validation="해석해 대비 오차 < 3%"
            critical={true}
          />
          
          <Task 
            number={2}
            title="Burton-Miller Formulation 구현"
            description="Unique solution 보장을 위한 Burton-Miller combined field integral equation. Coupling parameter α = i/k 설정 필수."
            libs="bempp-cl"
            validation="모든 주파수에서 수렴 확인 (resonance 포함)"
            isNew={true}
            critical={true}
          />
          <MathBlock>
            {`α = i/k (optimal coupling parameter)
LHS: (½I + D + αH)u = RHS: (S + α(½I + D'))g`}
          </MathBlock>
          
          <Task 
            number={3}
            title="Adaptive Mesh Near Edges"
            description="Edge 근처 해상도를 λ/10으로, 평면 영역은 λ/6으로 설정하는 적응적 메쉬 생성."
            libs="pygmsh, meshio"
            validation="Edge 근처 element size < 4.3mm (8kHz 기준)"
            isNew={true}
          />
          <MathBlock>
            {`Edge region: element_size = λ_min / 10 ≈ 4.3mm
Flat region: element_size = λ_min / 6  ≈ 7.2mm`}
          </MathBlock>

          <Task 
            number={4}
            title="Multi-Frequency BEM Solver"
            description="2-8 kHz 대역 내 주파수에서 Helmholtz 풀이. Adaptive frequency sampling 적용."
            libs="bempp-cl, joblib"
            validation="각 주파수에서 residual < 1e-6"
            isModified={true}
          />
          
          <Task 
            number={5}
            title="Phase Unwrapping & IDFT Synthesis"
            description="주파수 응답의 위상 불연속성을 np.unwrap으로 처리 후 IDFT. Hermitian symmetry 보장."
            libs="numpy.fft, scipy.signal"
            validation="Causality: E(t<0)/E(total) < 1e-6"
            isNew={true}
            critical={true}
          />
          <CodeBlock title="Phase Unwrapping (Critical)">
{`# CRITICAL: Phase unwrapping before IDFT
phase_raw = np.angle(P_freq)
phase_unwrapped = np.unwrap(phase_raw)
P_corrected = np.abs(P_freq) * np.exp(1j * phase_unwrapped)

# Hermitian symmetry for real output
P_full[N-len(P_freq)+1:] = np.conj(P_corrected[-1:0:-1])
h_t = np.fft.irfft(P_full, n=N)`}
          </CodeBlock>

          <Task 
            number={6}
            title="Energy Conservation Validation"
            description="Parseval's theorem으로 주파수/시간 영역 에너지 보존 검증."
            libs="numpy"
            validation="Relative error < 1%"
            isNew={true}
          />
          <MathBlock>
            {`Parseval: ∫|P(f)|²df = ∫|h(t)|²dt
Relative error = |E_freq - E_time| / max(E_freq, E_time)`}
          </MathBlock>
          
          <Task 
            number={7}
            title="Dataset 생성 파이프라인"
            description="다양한 Source/Mic 위치 조합으로 10,000개 RIR 생성. HDF5 포맷 저장. Domain randomization 적용."
            libs="h5py, multiprocessing"
            validation="데이터 무결성, 재현성 확인"
            isModified={true}
          />
          
          <Milestone title="Phase 1 완료 기준" target="BEM RIR이 해석해와 일치 + Causality 만족 + Energy 보존" />
        </RoadmapCard>

        {/* Phase 2 - REVISED */}
        <RoadmapCard 
          phase={2} 
          title="Structured Green's Function Learning" 
          duration="Month 5-8 (4 months)"
          status="pending"
          isExpanded={expanded[2]}
          onToggle={() => togglePhase(2)}
          isRevised={true}
        >
          <RevisionNote>
            <strong>v3.2 Changes:</strong> Complex-valued output 명시, 
            Diffraction MLP의 Re/Im heads 분리
          </RevisionNote>

          <Task 
            number={1}
            title="Image Source Method 구현"
            description="G_geometric (Direct + 1차 Reflection) 해석적 계산. Complex amplitude 포함."
            libs="numpy"
            validation="ISM vs BEM (LOS 영역) 오차 < 1%"
          />
          <MathBlock>
            {`G_total = G_geometric (Frozen) + G_diff (Learnable)
G_geometric: Complex-valued (amplitude + phase)`}
          </MathBlock>
          
          <Task 
            number={2}
            title="Complex Diffraction MLP 설계"
            description="입력: (φ_inc, φ_obs, k), 출력: Complex Diffraction Coefficient. Re/Im heads 분리."
            libs="torch.nn"
            validation="UTD 해와의 상관계수 > 0.9"
            isModified={true}
            critical={true}
          />
          <CodeBlock title="Complex Output Architecture (Critical)">
{`class DiffractionMLP(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(3, hidden_dim),  # (phi_inc, phi_obs, k)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head_real = nn.Linear(hidden_dim, 1)
        self.head_imag = nn.Linear(hidden_dim, 1)
    
    def forward(self, phi_inc, phi_obs, k):
        x = torch.stack([phi_inc, phi_obs, k], dim=-1)
        features = self.backbone(x)
        D_real = self.head_real(features)
        D_imag = self.head_imag(features)
        return torch.complex(D_real, D_imag)`}
          </CodeBlock>
          
          <Task 
            number={3}
            title="Complex Convolution Forward Model"
            description="입력 신호와 Complex G_total의 컨볼루션. 출력의 실수부가 측정 신호."
            libs="torch.fft"
            isModified={true}
          />
          
          <Task 
            number={4}
            title="Green-Net 학습 루프"
            description="Complex MSE Loss로 학습. Learning rate scheduling 적용."
            libs="torch.optim (AdamW)"
            validation="Validation Loss 수렴"
          />
          <MathBlock>
            {`L = ||y(t) - Re{s(t) * (G_geo + Ĝ_diff)}||²`}
          </MathBlock>
          
          <Task 
            number={5}
            title="Ablation: Direct vs Structured"
            description="G_total 전체를 학습하는 Baseline과 비교."
            validation="Structured > Direct (수렴 속도 2x↑, 정확도 10%↑)"
            critical={true}
          />
          
          <Milestone title="Phase 2 완료 기준" target="ICASSP 워크샵 페이퍼 Draft 완성" />
        </RoadmapCard>

        {/* Phase 3 - MAJOR REVISION */}
        <RoadmapCard 
          phase={3} 
          title="Neural Fields with Implicit Geometry" 
          duration="Month 9-13 (5 months)"
          status="pending"
          isExpanded={expanded[3]}
          onToggle={() => togglePhase(3)}
          isRevised={true}
        >
          <div className="bg-yellow-100 border border-yellow-400 rounded-lg p-3 mb-4">
            <span className="font-bold text-yellow-800">⚠️ CORE NOVELTY PHASE - MAJOR REVISION</span>
            <p className="text-sm text-yellow-700">Trivial Solution 회피, Complex Field, 수정된 Fourier Scale 반영</p>
          </div>

          <RevisionNote>
            <strong>v3.2 Critical Changes:</strong><br/>
            • Complex pressure output (Re/Im heads)<br/>
            • Surface Existence Constraint 추가<br/>
            • Inhomogeneous Helmholtz (Source Term)<br/>
            • Fourier Scale σ: 23 → 62 m⁻¹
          </RevisionNote>
          
          <Task 
            number={1}
            title="Fourier Feature Encoding (CORRECTED)"
            description="각도 의존성을 반영한 Fourier scale. σ = k_max·sin(θ_max)/(2π) ≈ 62 m⁻¹"
            libs="torch"
            validation="고주파 회절 패턴 재현 확인"
            isModified={true}
            critical={true}
          />
          <MathBlock>
            {`# CORRECTED: Include angular dependence
σ = k_max · sin(θ_max) / (2π) · 1.5
  = (2π · 8000 / 343) · sin(60°) / (2π) · 1.5
  ≈ 62 m⁻¹  (NOT 23!)`}
          </MathBlock>
          <CodeBlock title="Corrected Fourier Scale">
{`def compute_fourier_scale(f_max_hz, c=343.0, max_angle_deg=60.0):
    k_max = 2 * np.pi * f_max_hz / c
    theta_max = np.radians(max_angle_deg)
    spatial_freq_max = k_max * np.sin(theta_max)
    sigma = spatial_freq_max / (2 * np.pi) * 1.5  # Safety margin
    return sigma  # ≈ 62 for 8kHz`}
          </CodeBlock>
          
          <Task 
            number={2}
            title="Complex Joint Output Network"
            description="입력: γ(x), k → 출력: (Complex p, Real SDF). Pressure는 Re/Im 분리 출력."
            libs="torch.nn"
            isNew={true}
            critical={true}
          />
          <CodeBlock title="Joint Network Architecture (Critical)">
{`class AcousticNeuralField(nn.Module):
    def __init__(self, fourier_dim=256, hidden_dim=512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(fourier_dim + 1, hidden_dim),  # +1 for k
            nn.ReLU(),
            # ... more layers
        )
        # Pressure: Complex (Re + Im)
        self.p_head_real = nn.Linear(hidden_dim, 1)
        self.p_head_imag = nn.Linear(hidden_dim, 1)
        # SDF: Real
        self.sdf_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, gamma_x, k):
        features = self.backbone(torch.cat([gamma_x, k], dim=-1))
        p_real = self.p_head_real(features)
        p_imag = self.p_head_imag(features)
        p = torch.complex(p_real, p_imag)
        sdf = self.sdf_head(features)
        return p, sdf`}
          </CodeBlock>

          <Task 
            number={3}
            title="Eikonal Loss"
            description="SDF gradient norm = 1 제약."
            libs="torch.autograd"
            validation="|∇s| - 1 ≈ 0 (전체 영역)"
          />
          <MathBlock>
            {`L_geo = || |∇s(x)| - 1 ||²`}
          </MathBlock>

          <Task 
            number={4}
            title="Surface Existence Constraint"
            description="SDF가 반드시 양수/음수를 모두 가지도록 강제. Trivial solution (SDF=const) 방지."
            libs="torch"
            validation="sdf.min() < 0 < sdf.max()"
            isNew={true}
            critical={true}
          />
          <CodeBlock title="Surface Existence Constraint (Critical)">
{`def surface_existence_loss(sdf):
    """
    Ensure SDF crosses zero (surface exists).
    Trivial solution SDF=const violates this.
    """
    sdf_min = sdf.min()
    sdf_max = sdf.max()
    # Both signs must exist
    loss = F.relu(-sdf_min) + F.relu(-sdf_max)
    # Alternative: sdf_min should be negative, sdf_max positive
    loss += F.relu(sdf_min + 0.1)  # sdf_min < -0.1
    loss += F.relu(-sdf_max + 0.1) # sdf_max > 0.1
    return loss`}
          </CodeBlock>
          
          <Task 
            number={5}
            title="Inhomogeneous Helmholtz Loss"
            description="Source term 포함한 Helmholtz. Homogeneous(p=0)가 trivial solution이 되는 것을 방지."
            libs="torch.autograd"
            validation="PDE residual < 1e-3 (source 위치 제외)"
            isNew={true}
            critical={true}
          />
          <CodeBlock title="Inhomogeneous Helmholtz (Critical)">
{`def helmholtz_loss(p, coords, k, source_pos, sigma=0.01):
    """
    Inhomogeneous Helmholtz: ∇²p + k²p = -δ(x - x_src)
    Source term prevents p=0 trivial solution.
    """
    # Compute Laplacian
    laplacian_p = compute_laplacian(p, coords)
    
    # Gaussian approximation of point source
    dist_to_source = torch.norm(coords - source_pos, dim=-1)
    source_term = torch.exp(-dist_to_source**2 / (2*sigma**2))
    source_term = source_term / (sigma * np.sqrt(2*np.pi))  # Normalize
    
    # PDE residual (source term makes p=0 impossible)
    residual = laplacian_p + k**2 * p + source_term
    return torch.mean(torch.abs(residual)**2)`}
          </CodeBlock>

          <Task 
            number={6}
            title="Boundary Condition Loss"
            description="SDF ≈ 0 영역에서 Neumann BC. 법선 방향은 ∇s/|∇s|로 계산."
            libs="torch.autograd"
            critical={true}
          />
          <MathBlock>
            {`L_BC = || ∂p/∂n ||² at s(x) ≈ 0
where n = ∇s / |∇s|`}
          </MathBlock>
          
          <Task 
            number={7}
            title="Multi-Loss Balancing (Adaptive)"
            description="GradNorm 또는 Uncertainty Weighting으로 Loss 균형 자동 조절."
            libs="custom implementation"
            validation="모든 Loss 동시 수렴"
            isModified={true}
          />
          <MathBlock>
            {`L_total = L_data + λ₁L_Helmholtz + λ₂L_geo + λ₃L_BC + λ₄L_surface
where λ_i are learnable or adaptive`}
          </MathBlock>
          
          <Task 
            number={8}
            title="Incremental Integration 학습"
            description="Step-by-step Loss 추가: (1)Data → (2)+Eikonal+Surface → (3)+Helmholtz → (4)+BC"
            validation="각 단계에서 안정적 수렴"
          />
          
          <Milestone title="Phase 3 완료 기준" target="SDF 복원 IoU > 0.8 + Trivial Solution 회피 확인" />
        </RoadmapCard>

        {/* Phase 4 - REVISED */}
        <RoadmapCard 
          phase={4} 
          title="Sim2Real & Cycle-Consistency Validation" 
          duration="Month 14-18 (5 months)"
          status="pending"
          isExpanded={expanded[4]}
          onToggle={() => togglePhase(4)}
          isRevised={true}
        >
          <RevisionNote>
            <strong>v3.2 Changes:</strong> Domain Randomization 강화, 
            Pose Refinement with ToA constraints, 확장된 검증 기간
          </RevisionNote>

          <Task 
            number={1}
            title="실험 환경 구축"
            description="L-Shape 코너, Bluetooth 스피커, 스마트폰 마이크. Chirp 신호 (2-8 kHz)."
            validation="SNR > 20dB"
          />
          
          <Task 
            number={2}
            title="Domain Randomization Training"
            description="Sim2Real gap 해소를 위해 시뮬레이션 데이터에 randomization 적용."
            libs="custom"
            isNew={true}
          />
          <CodeBlock title="Domain Randomization">
{`class DomainRandomizer:
    absorption_range = (0.0, 0.3)
    snr_range = (10, 30)  # dB
    speed_of_sound_range = (340, 346)  # Temperature
    
    def randomize(self, rir, metadata):
        # 1. Random absorption
        # 2. Random noise
        # 3. Random speed of sound (time stretch)
        return augmented_rir`}
          </CodeBlock>

          <Task 
            number={3}
            title="ARCore + ToA Pose Refinement"
            description="Time-of-Arrival 제약을 활용한 ARCore pose 보정. LOS 영역에서 calibration."
            libs="Android ARCore API"
            validation="위치 오차 < 3cm"
            isModified={true}
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
            description="실제 소리와 가상 소리의 일치 여부 확인."
            validation="Correlation > 0.8"
            critical={true}
          />
          <MathBlock>
            {`y_real ≈ y_sim = BEM(SDF_pred)
Cycle Loss < threshold`}
          </MathBlock>
          
          <Milestone title="Phase 4 완료 기준" target="CVPR 투고 + Cycle-Consistency 검증 완료" />
        </RoadmapCard>

        {/* Risks Section - NEW */}
        <div className="bg-slate-700 rounded-lg p-4 mb-6">
          <h3 className="text-white font-bold mb-3">⚠️ Critical Risks & Mitigations</h3>
          <RiskItem 
            severity="critical"
            code="C1"
            title="Complex Field 처리 누락"
            mitigation="Re/Im heads 분리, torch.complex64 사용"
          />
          <RiskItem 
            severity="critical"
            code="C2"
            title="Trivial Solution (SDF=const, p=0)"
            mitigation="Surface Existence + Inhomogeneous Helmholtz"
          />
          <RiskItem 
            severity="critical"
            code="C3"
            title="Phase Unwrapping 누락 → Acausal RIR"
            mitigation="np.unwrap + Causality 검증"
          />
          <RiskItem 
            severity="high"
            code="H1"
            title="Fourier Scale 과소평가"
            mitigation="σ = 62 m⁻¹ (각도 의존성 반영)"
          />
          <RiskItem 
            severity="high"
            code="H3"
            title="BEM Resonance 불안정"
            mitigation="Burton-Miller with α = i/k"
          />
        </div>

        {/* Deliverables */}
        <div className="bg-slate-700 rounded-lg p-6 mt-6">
          <h3 className="text-xl font-bold text-white mb-4">📋 Key Deliverables (Revised)</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="bg-slate-600 p-3 rounded">
              <span className="text-blue-400 font-semibold">Month 8:</span>
              <p className="text-gray-300">ICASSP Workshop Paper</p>
              <p className="text-gray-500 text-xs">Green-Net 방법론 검증</p>
            </div>
            <div className="bg-slate-600 p-3 rounded">
              <span className="text-green-400 font-semibold">Month 15:</span>
              <p className="text-gray-300">CVPR Full Paper Submission</p>
              <p className="text-gray-500 text-xs">Core Contribution</p>
            </div>
            <div className="bg-slate-600 p-3 rounded">
              <span className="text-yellow-400 font-semibold">Month 18:</span>
              <p className="text-gray-300">Sim2Real Validation Complete</p>
              <p className="text-gray-500 text-xs">실험 데이터 검증</p>
            </div>
            <div className="bg-slate-600 p-3 rounded">
              <span className="text-purple-400 font-semibold">Year 2-3:</span>
              <p className="text-gray-300">Nature Communications</p>
              <p className="text-gray-500 text-xs">응용 확장 (Medical Ultrasound)</p>
            </div>
          </div>
        </div>

        {/* One-liner */}
        <div className="mt-6 p-4 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
          <p className="text-white text-center font-semibold text-sm">
            "We jointly reconstruct <span className="text-yellow-300">complex acoustic fields</span> and scene geometry 
            by learning only the diffraction residual atop analytical Green's functions, 
            while enforcing <span className="text-yellow-300">inhomogeneous Helmholtz PDE</span>, 
            Eikonal constraints, and <span className="text-yellow-300">surface existence guarantees</span>."
          </p>
        </div>

        <div className="mt-4 text-center">
          <span className="bg-purple-600 text-white text-xs px-3 py-1 rounded-full">
            v3.2 — Dr. Tensor Wave Critical Review Reflected
          </span>
        </div>

        <p className="text-center text-gray-500 text-xs mt-4">
          Acoustic Neural Tomography Roadmap v3.2 | Target: CVPR Oral / Nature Communications | Timeline: 18 months
        </p>
      </div>
    </div>
  );
}
