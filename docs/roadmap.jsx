import { useState } from "react";

const PHASES = [
  {
    id: 0,
    phase: "Phase 0",
    title: "기반 검증 (Foundation Validation)",
    period: "Month 1-2",
    color: "#f59e0b",
    icon: "⚡",
    status: "START HERE",
    objectives: [
      "Bempp-cl 환경 구축 및 BEM 파이프라인 검증",
      "Infinite Wedge 해석해 vs BEM 오차 < 3% 달성",
      "GWA / SoundSpaces 2.0 데이터셋 다운로드 및 탐색",
    ],
    math: {
      title: "검증 기준: Macdonald의 Wedge Diffraction 해석해",
      equations: [
        "G_wedge = G₀ + G_diff",
        "G_diff ∝ (e^{ikr}/√r) · D(φ, φ', n)",
        "BEM 오차 = ‖p_BEM − p_analytical‖₂ / ‖p_analytical‖₂ < 0.03",
      ],
    },
    tasks: [
      { task: "pip install bempp-cl, meshio, pygmsh", effort: "1일", critical: false },
      { task: "Infinite Wedge 메쉬 생성 (N < 10,000)", effort: "3일", critical: true },
      { task: "단일 주파수 (f=2kHz) Helmholtz BEM solve", effort: "3일", critical: true },
      { task: "해석해 비교 → 오차율 리포트", effort: "2일", critical: true },
      { task: "GWA pre-computed IR 다운로드 (subset)", effort: "2일", critical: false },
      { task: "SoundSpaces 2.0 설치 + Replica NLOS 시나리오 추출", effort: "5일", critical: false },
    ],
    deliverable: "BEM vs 해석해 비교 리포트 (오차 < 3%)",
    risk: "Bempp-cl OpenCL 호환성 문제 → 백업: pyroomacoustics",
  },
  {
    id: 1,
    phase: "Phase 1",
    title: "BEM 합성 데이터 생성 (Data Factory)",
    period: "Month 2-4",
    color: "#3b82f6",
    icon: "🔧",
    status: "DATA",
    objectives: [
      "NLOS 회절 특화 합성 데이터셋 구축 (15장면 × 75쌍 × 30freq)",
      "Fresnel Zone 분석 기반 주파수 선정",
      "GWA 데이터와 BEM 데이터 교차 검증",
    ],
    math: {
      title: "Fresnel Number 기반 주파수 최적화",
      equations: [
        "F = a² / (λL), λ = c/f",
        "F ≈ 1 → 회절 극대화 영역",
        "f_optimal = c·a² / L  (a: 장애물, L: 관측거리)",
        "IDFT 합성: h(t) = IFFT{P(f₁), P(f₂), ..., P(fₙ)}",
      ],
    },
    tasks: [
      { task: "L-shape/T-junction/U-shape 메쉬 자동 생성 파이프라인", effort: "1주", critical: true },
      { task: "Multi-freq BEM sweep (30 freq × 15 scene)", effort: "1-2주 (CPU)", critical: true },
      { task: "IDFT로 시간영역 RIR 합성 + Causality 검증", effort: "3일", critical: true },
      { task: "Fresnel Zone 분석 → Shadow/Lit/Transition 영역 라벨링", effort: "3일", critical: false },
      { task: "GWA NLOS 장면 추출 + BEM 비교 → 신뢰 주파수 대역 확인", effort: "1주", critical: false },
      { task: "SDF Ground Truth 생성 (메쉬 → Signed Distance Field)", effort: "3일", critical: true },
    ],
    deliverable: "NLOS Acoustic Diffraction Dataset v1.0 (공개 시 추가 contribution)",
    risk: "BEM 계산 시간 초과 → 주파수/장면 수 축소 또는 적응적 샘플링",
  },
  {
    id: 2,
    phase: "Phase 2",
    title: "Forward Model — Structured Green Learning",
    period: "Month 4-7",
    color: "#8b5cf6",
    icon: "🧠",
    status: "CORE",
    objectives: [
      "Structured Green's Function 학습: G₀ + G_ref (고정) + MLP_diff (학습)",
      "Fourier Features + SIREN으로 Spectral Bias 해결",
      "Helmholtz PDE Loss 통합 및 수렴 확인",
    ],
    math: {
      title: "Structured Green's Function + PINN",
      equations: [
        "Ĝ_total = G₀ (Direct, frozen) + G_ref (Reflection, frozen) + MLP_θ(φ,φ',k,L) (Diffraction, learnable)",
        "입력 변환: γ(x) = [cos(2πBx), sin(2πBx)], σ ≈ f_max/c",
        "L_total = L_data + λ₁·L_Helmholtz + λ₂·L_geo",
        "L_Helmholtz = ‖∇²p̂ + k²p̂‖²",
        "L_geo = ‖ ‖∇s‖ − 1 ‖² (Eikonal)",
        "SIREN: sin(ω₀ · Wx + b), ω₀ ∝ k",
      ],
    },
    tasks: [
      { task: "SIREN (6층×512) + Fourier Features (128dim) 구현", effort: "1주", critical: true },
      { task: "단일 장면 Forward fitting (p 출력만)", effort: "1주", critical: true },
      { task: "Helmholtz PDE Loss 추가 (torch.autograd 2차미분)", effort: "1주", critical: true },
      { task: "Structured Green: G₀+G_ref 해석적 고정 + MLP_diff 학습", effort: "2주", critical: true },
      { task: "Multi-scale training (저주파→고주파 curriculum)", effort: "1주", critical: false },
      { task: "Adaptive Loss Weighting (GradNorm 또는 동적 λ)", effort: "3일", critical: false },
      { task: "15장면 전체 학습 + 수렴 분석", effort: "1주", critical: true },
    ],
    deliverable: "Forward model: BEM 대비 재구성 오차 < 5% (NLOS 포함)",
    risk: "Spectral Bias 잔존 → ω₀ 스케줄링 조정. VRAM 부족 → gradient checkpointing 필수",
  },
  {
    id: 3,
    phase: "Phase 3",
    title: "Inverse Model — Sound → Geometry",
    period: "Month 7-10",
    color: "#ec4899",
    icon: "🔮",
    status: "NOVELTY",
    objectives: [
      "SDF 동시 출력: f_θ(γ(x), audio features) → (p, SDF)",
      "음향 신호만으로 NLOS 기하구조 복원",
      "경계조건 Loss (L_BC) 통합",
    ],
    math: {
      title: "Inverse Problem: 소리 → 기하구조",
      equations: [
        "f_θ: (γ(x), t) → (p̂, ŝ)  [p: 압력, s: SDF]",
        "L_BC = Σ_{s(x)≈0} |∂p/∂n + ikβ·p|²  (Robin BC)",
        "법선 벡터: n = ∇s / ‖∇s‖  (SDF gradient에서 자동 추출)",
        "L_total = L_data + λ₁L_Helmholtz + λ₂L_Eikonal + λ₃L_BC",
        "SDF=0 등위면 → Marching Cubes → 3D 기하구조 복원",
      ],
    },
    tasks: [
      { task: "SDF head 추가 (p, s 동시 출력)", effort: "1주", critical: true },
      { task: "Eikonal Loss ‖∇s‖=1 구현 및 안정화", effort: "3일", critical: true },
      { task: "BC Loss 구현 (SDF ≈ 0 영역 자동 검출)", effort: "1주", critical: true },
      { task: "Alternating Training: Forward head ↔ SDF head 교대", effort: "1주", critical: true },
      { task: "단일 장면 SDF 복원 → GT SDF와 Chamfer Distance 비교", effort: "1주", critical: true },
      { task: "15장면 전체 → SDF 복원 성능 통계", effort: "2주", critical: true },
      { task: "Marching Cubes로 메쉬 추출 → 시각화", effort: "3일", critical: false },
    ],
    deliverable: "Monaural audio → NLOS 기하구조 복원 (Chamfer Distance 정량화)",
    risk: "Joint Learning 수렴 실패 → 교대 학습 + Loss 밸런싱. SDF 품질 저하 → pre-train Eikonal 단독",
  },
  {
    id: 4,
    phase: "Phase 4",
    title: "검증 & 일반화 (Validation & Generalization)",
    period: "Month 10-14",
    color: "#10b981",
    icon: "📊",
    status: "PROOF",
    objectives: [
      "Cycle-Consistency 검증 (Inverse → Forward → 비교)",
      "GWA/SoundSpaces 대규모 장면 일반화 테스트",
      "RAF 실측 데이터 sim-to-real gap 정량화",
      "Ablation Study 4종 완료",
    ],
    math: {
      title: "Cycle-Consistency & Ablation",
      equations: [
        "Cycle: audio_real → [Inverse] → SDF → [Forward Surrogate] → audio_synth",
        "L_cycle = ‖audio_real − audio_synth‖²",
        "Ablation A: Full MLP (no Structured Green) vs Ours",
        "Ablation B: GT geometry vs SDF output",
        "Ablation C: No Helmholtz Loss vs Ours",
        "Ablation D: No Fourier Features vs Ours",
      ],
    },
    tasks: [
      { task: "Forward Surrogate (경량 네트워크) 학습 → Cycle에 사용", effort: "1주", critical: true },
      { task: "Cycle-Consistency 파이프라인 구현", effort: "1주", critical: true },
      { task: "GWA 복잡 장면 (가구 포함) 테스트", effort: "2주", critical: true },
      { task: "SoundSpaces Replica 환경 테스트", effort: "1주", critical: false },
      { task: "RAF 실측 데이터 비교 (sim-to-real gap)", effort: "1주", critical: true },
      { task: "Ablation Study 4종 실행 + 표/그래프", effort: "2주", critical: true },
      { task: "Baseline 비교: NAF, MESH2IR, pyroomacoustics", effort: "1주", critical: true },
    ],
    deliverable: "완전한 실험 결과 + Ablation + Baseline 비교",
    risk: "일반화 실패 → 학습 데이터 augmentation (geometry perturbation). Sim-to-real gap 과대 → domain adaptation",
  },
  {
    id: 5,
    phase: "Phase 5",
    title: "논문 작성 & 투고 (Writing & Submission)",
    period: "Month 14-18",
    color: "#f43f5e",
    icon: "📝",
    status: "PUBLISH",
    objectives: [
      "Year 1: ICASSP / WASPAA 워크샵 페이퍼 (방법론 검증)",
      "Year 2: CVPR / ECCV / NeurIPS 풀 페이퍼",
      "Year 3: Nature Communications / TPAMI 저널",
    ],
    math: {
      title: "One-Line Contribution (외워라)",
      equations: [
        "\"We propose the first physics-rigorous framework that jointly reconstructs acoustic fields and scene geometry from monaural audio by learning only the diffraction residual atop analytical Green's functions, while enforcing Helmholtz PDE and Eikonal constraints.\"",
      ],
    },
    tasks: [
      { task: "ICASSP 4p 논문 초고 (Phase 0-2 결과)", effort: "3주", critical: true },
      { task: "CVPR 풀페이퍼 초고 (전체 파이프라인)", effort: "6주", critical: true },
      { task: "시각화: NLOS 복원 영상, SDF 진화 GIF", effort: "1주", critical: false },
      { task: "Rebuttal 대비 추가 실험 버퍼", effort: "2주", critical: false },
      { task: "코드/데이터 공개 준비 (GitHub + 데이터셋 호스팅)", effort: "1주", critical: false },
    ],
    deliverable: "투고 완료 + 코드/데이터 공개",
    risk: "리뷰어: 'venue mismatch' → Framing을 'Seeing Around Corners with Sound'로 강화",
  },
];

const HARDWARE = {
  cpu: { name: "i9-9900K", spec: "8C/16T, 3.6GHz", limit: "BEM solve: 가능. 병렬 sweep에 반나절", ok: true },
  ram: { name: "32GB DDR4", spec: "32,768 MB", limit: "BEM 메쉬 N < 20,000 요소. 복잡 3D 환경 불가", ok: true },
  gpu: { name: "RTX 2080 Super", spec: "VRAM 8GB (실제)", limit: "SIREN 6×512 + FF128 한계. 2차미분 메모리 2-3배", ok: false },
};

const DATA_TIERS = [
  {
    tier: "Tier 1",
    name: "오픈 데이터셋",
    purpose: "기초 체력 + 일반화",
    color: "#3b82f6",
    datasets: [
      { name: "GWA", source: "UMD (SIGGRAPH'22)", size: "200만 RIR", value: "FDTD+Ray hybrid, 저주파 회절 포함, 3D mesh GT", warning: "재시뮬레이션은 HPC 필요. pre-computed IR만 사용" },
      { name: "SoundSpaces 2.0", source: "Meta (NeurIPS'22)", size: "Matterport3D+Replica", value: "RGB-D 이미지 포함 → Vision 연결 가능", warning: "회절 모델링 부정확 (ray-tracing 기반)" },
      { name: "RAF", source: "Google (2024)", size: "실측 RIR + 이미지 + 6DoF", value: "실제 데이터! Sim-to-real gap 검증", warning: "장면 수 제한적" },
      { name: "dEchorate", source: "Bar-Ilan (2021)", size: "1,800 RIR", value: "에코 타이밍 정밀 annotation, 벽 구성 변경", warning: "회절 annotation 없음" },
    ],
  },
  {
    tier: "Tier 2",
    name: "자체 BEM 합성",
    purpose: "정밀 타격 (회절 특화)",
    color: "#8b5cf6",
    datasets: [
      { name: "NLOS Diffraction Dataset", source: "자체 생성 (Bempp-cl)", size: "~33,750 BEM solve", value: "회절 지배적 NLOS의 정밀 GT. 공개 시 추가 contribution", warning: "2D/2.5D 한계. 복잡 3D 불가 (RAM 제약)" },
    ],
  },
  {
    tier: "Tier 3",
    name: "해석해 (Analytical)",
    purpose: "이론적 앵커 + BEM 검증",
    color: "#f59e0b",
    datasets: [
      { name: "Infinite Wedge", source: "Macdonald (1915)", size: "해석적", value: "BEM 검증의 gold standard", warning: "-" },
      { name: "Half-plane", source: "Sommerfeld (1896)", size: "해석적", value: "회절 원조 문제", warning: "-" },
      { name: "Circular Cylinder", source: "Bessel 급수", size: "해석적", value: "곡면 회절 검증", warning: "구현 복잡" },
    ],
  },
];

const Tab = ({ active, onClick, children, color }) => (
  <button
    onClick={onClick}
    style={{
      padding: "10px 20px",
      border: "none",
      borderBottom: active ? `3px solid ${color || "#8b5cf6"}` : "3px solid transparent",
      background: active ? "rgba(139,92,246,0.08)" : "transparent",
      color: active ? "#e2e8f0" : "#94a3b8",
      fontSize: "14px",
      fontWeight: active ? 700 : 500,
      cursor: "pointer",
      transition: "all 0.2s",
      fontFamily: "'JetBrains Mono', monospace",
      letterSpacing: "-0.02em",
    }}
  >
    {children}
  </button>
);

const Badge = ({ children, color }) => (
  <span
    style={{
      display: "inline-block",
      padding: "2px 10px",
      borderRadius: "4px",
      background: `${color}22`,
      color: color,
      fontSize: "11px",
      fontWeight: 700,
      fontFamily: "'JetBrains Mono', monospace",
      letterSpacing: "0.05em",
      border: `1px solid ${color}44`,
    }}
  >
    {children}
  </span>
);

export default function PhDRoadmap() {
  const [activeTab, setActiveTab] = useState("overview");
  const [expandedPhase, setExpandedPhase] = useState(0);
  const [showMath, setShowMath] = useState({});

  const toggleMath = (id) => setShowMath((p) => ({ ...p, [id]: !p[id] }));

  return (
    <div
      style={{
        fontFamily: "'IBM Plex Sans', 'Noto Sans KR', sans-serif",
        background: "#0a0e1a",
        color: "#e2e8f0",
        minHeight: "100vh",
        padding: "0",
      }}
    >
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet" />

      {/* Header */}
      <div
        style={{
          background: "linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)",
          borderBottom: "1px solid #1e293b",
          padding: "32px 32px 0",
        }}
      >
        <div style={{ maxWidth: 960, margin: "0 auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 4 }}>
            <span style={{ fontSize: 11, color: "#f59e0b", fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.1em", fontWeight: 700 }}>
              PhD RESEARCH ROADMAP v3.2 — FINAL
            </span>
          </div>
          <h1 style={{ fontSize: 26, fontWeight: 700, margin: "8px 0", lineHeight: 1.3, letterSpacing: "-0.03em" }}>
            <span style={{ color: "#a78bfa" }}>Seeing Around Corners with Sound</span>
          </h1>
          <p style={{ fontSize: 14, color: "#94a3b8", margin: "0 0 6px", lineHeight: 1.5, maxWidth: 700 }}>
            Monaural Non-Line-of-Sight 3D Reconstruction via Physics-Informed Neural Fields
          </p>
          <p style={{ fontSize: 12, color: "#64748b", margin: "0 0 20px", fontFamily: "'JetBrains Mono', monospace" }}>
            Target: ICASSP (Y1) → CVPR/ECCV (Y2) → Nature Comms (Y3) &nbsp;|&nbsp; Hardware: i9-9900K · 32GB · RTX 2080S (8GB)
          </p>

          {/* Tabs */}
          <div style={{ display: "flex", gap: 0, borderBottom: "1px solid #1e293b", marginBottom: -1 }}>
            {[
              ["overview", "📋 종합"],
              ["phases", "🗺️ Phase 상세"],
              ["data", "📂 데이터 전략"],
              ["hardware", "🖥️ 하드웨어"],
              ["risk", "⚠️ 리스크"],
            ].map(([key, label]) => (
              <Tab key={key} active={activeTab === key} onClick={() => setActiveTab(key)} color={key === "phases" ? "#8b5cf6" : key === "data" ? "#3b82f6" : key === "hardware" ? "#f59e0b" : key === "risk" ? "#f43f5e" : "#8b5cf6"}>
                {label}
              </Tab>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div style={{ maxWidth: 960, margin: "0 auto", padding: "24px 32px 48px" }}>
        {/* OVERVIEW TAB */}
        {activeTab === "overview" && (
          <div>
            {/* Timeline Bar */}
            <div style={{ marginBottom: 32 }}>
              <h3 style={{ fontSize: 14, color: "#94a3b8", fontWeight: 600, marginBottom: 16, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.05em" }}>
                TIMELINE — 18 MONTHS
              </h3>
              <div style={{ display: "flex", gap: 3, marginBottom: 8 }}>
                {PHASES.map((p) => (
                  <div
                    key={p.id}
                    style={{
                      flex: p.id === 4 ? 2 : p.id === 5 ? 2 : p.id === 2 ? 1.5 : p.id === 3 ? 1.5 : 1,
                      background: `${p.color}33`,
                      border: `1px solid ${p.color}66`,
                      borderRadius: 4,
                      padding: "8px 10px",
                      cursor: "pointer",
                      transition: "all 0.2s",
                    }}
                    onClick={() => { setActiveTab("phases"); setExpandedPhase(p.id); }}
                  >
                    <div style={{ fontSize: 10, color: p.color, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>{p.phase}</div>
                    <div style={{ fontSize: 11, color: "#cbd5e1", marginTop: 2 }}>{p.period}</div>
                  </div>
                ))}
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#475569", fontFamily: "'JetBrains Mono', monospace" }}>
                <span>M1</span><span>M4</span><span>M7</span><span>M10</span><span>M14</span><span>M18</span>
              </div>
            </div>

            {/* Core Architecture */}
            <div style={{ background: "#111827", border: "1px solid #1e293b", borderRadius: 8, padding: 24, marginBottom: 24 }}>
              <h3 style={{ fontSize: 14, color: "#a78bfa", fontWeight: 700, marginBottom: 16, fontFamily: "'JetBrains Mono', monospace" }}>
                핵심 아키텍처: Structured Green's Function + Implicit Geometry
              </h3>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13, lineHeight: 2, color: "#e2e8f0", background: "#0a0e1a", padding: 20, borderRadius: 6, border: "1px solid #1e293b", overflowX: "auto" }}>
                <div><span style={{ color: "#64748b" }}>// Forward Model</span></div>
                <div>Ĝ_total = <span style={{ color: "#22c55e" }}>G₀ (Direct)</span> + <span style={{ color: "#3b82f6" }}>G_ref (Reflection)</span> + <span style={{ color: "#f59e0b" }}>MLP_θ(φ,φ',k,L)</span></div>
                <div style={{ color: "#64748b" }}>{"           "}frozen{"            "}frozen{"            "}learnable (diffraction)</div>
                <div style={{ marginTop: 8 }}><span style={{ color: "#64748b" }}>// Inverse Model</span></div>
                <div>f_θ: (<span style={{ color: "#a78bfa" }}>γ(x)</span>, t) → (<span style={{ color: "#ec4899" }}>p̂</span>, <span style={{ color: "#f59e0b" }}>ŝ</span>){"   "}<span style={{ color: "#64748b" }}>// p: 음압, s: SDF</span></div>
                <div style={{ marginTop: 8 }}><span style={{ color: "#64748b" }}>// Loss</span></div>
                <div>L = L_data + λ₁·<span style={{ color: "#3b82f6" }}>L_Helmholtz</span> + λ₂·<span style={{ color: "#f59e0b" }}>L_Eikonal</span> + λ₃·<span style={{ color: "#ec4899" }}>L_BC</span></div>
                <div style={{ marginTop: 8 }}><span style={{ color: "#64748b" }}>// Cycle-Consistency</span></div>
                <div>audio → <span style={{ color: "#ec4899" }}>[Inverse]</span> → SDF → <span style={{ color: "#22c55e" }}>[Forward]</span> → audio' ≈ audio</div>
              </div>
            </div>

            {/* One-Line Contribution */}
            <div style={{ background: "linear-gradient(135deg, #1e1b4b, #172554)", border: "1px solid #a78bfa44", borderRadius: 8, padding: 24, marginBottom: 24 }}>
              <div style={{ fontSize: 11, color: "#a78bfa", fontWeight: 700, marginBottom: 8, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.08em" }}>
                ONE-LINE CONTRIBUTION — 외워라
              </div>
              <p style={{ fontSize: 14, color: "#e2e8f0", lineHeight: 1.7, margin: 0, fontStyle: "italic" }}>
                "We propose the first physics-rigorous framework that jointly reconstructs acoustic fields and scene geometry from monaural audio by learning only the diffraction residual atop analytical Green's functions, while enforcing Helmholtz PDE and Eikonal constraints."
              </p>
            </div>

            {/* Key Numbers */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
              {[
                { label: "BEM 장면", value: "15", unit: "scenes", color: "#3b82f6" },
                { label: "BEM solve 총", value: "~34K", unit: "solves", color: "#8b5cf6" },
                { label: "SIREN 규모", value: "6×512", unit: "~1.5M params", color: "#f59e0b" },
                { label: "목표 기간", value: "18", unit: "months", color: "#f43f5e" },
              ].map((item, i) => (
                <div key={i} style={{ background: "#111827", border: "1px solid #1e293b", borderRadius: 8, padding: "16px 12px", textAlign: "center" }}>
                  <div style={{ fontSize: 28, fontWeight: 700, color: item.color, fontFamily: "'JetBrains Mono', monospace" }}>{item.value}</div>
                  <div style={{ fontSize: 11, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>{item.unit}</div>
                  <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 4 }}>{item.label}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* PHASES TAB */}
        {activeTab === "phases" && (
          <div>
            {PHASES.map((phase) => {
              const isExpanded = expandedPhase === phase.id;
              return (
                <div
                  key={phase.id}
                  style={{
                    background: isExpanded ? "#111827" : "#0d1117",
                    border: `1px solid ${isExpanded ? phase.color + "66" : "#1e293b"}`,
                    borderRadius: 8,
                    marginBottom: 12,
                    overflow: "hidden",
                    transition: "all 0.2s",
                  }}
                >
                  {/* Phase Header */}
                  <div
                    onClick={() => setExpandedPhase(isExpanded ? -1 : phase.id)}
                    style={{ padding: "16px 20px", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "space-between" }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      <span style={{ fontSize: 20 }}>{phase.icon}</span>
                      <div>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <span style={{ fontSize: 12, color: phase.color, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>
                            {phase.phase}
                          </span>
                          <Badge color={phase.color}>{phase.status}</Badge>
                        </div>
                        <div style={{ fontSize: 15, fontWeight: 600, marginTop: 2 }}>{phase.title}</div>
                      </div>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      <span style={{ fontSize: 12, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>{phase.period}</span>
                      <span style={{ color: "#475569", fontSize: 18, transform: isExpanded ? "rotate(180deg)" : "rotate(0)", transition: "transform 0.2s" }}>▾</span>
                    </div>
                  </div>

                  {/* Expanded Content */}
                  {isExpanded && (
                    <div style={{ padding: "0 20px 20px", borderTop: "1px solid #1e293b" }}>
                      {/* Objectives */}
                      <div style={{ marginTop: 16, marginBottom: 16 }}>
                        <div style={{ fontSize: 11, color: "#64748b", fontWeight: 700, marginBottom: 8, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.08em" }}>OBJECTIVES</div>
                        {phase.objectives.map((obj, i) => (
                          <div key={i} style={{ fontSize: 13, color: "#cbd5e1", padding: "4px 0", paddingLeft: 16, borderLeft: `2px solid ${phase.color}44`, marginBottom: 4 }}>
                            {obj}
                          </div>
                        ))}
                      </div>

                      {/* Math Toggle */}
                      <div style={{ marginBottom: 16 }}>
                        <button
                          onClick={(e) => { e.stopPropagation(); toggleMath(phase.id); }}
                          style={{
                            background: showMath[phase.id] ? `${phase.color}22` : "transparent",
                            border: `1px solid ${phase.color}44`,
                            borderRadius: 4,
                            padding: "6px 14px",
                            color: phase.color,
                            fontSize: 12,
                            fontWeight: 600,
                            cursor: "pointer",
                            fontFamily: "'JetBrains Mono', monospace",
                          }}
                        >
                          {showMath[phase.id] ? "▾" : "▸"} Math & Physics
                        </button>
                        {showMath[phase.id] && (
                          <div style={{ background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 6, padding: 16, marginTop: 8 }}>
                            <div style={{ fontSize: 12, color: phase.color, fontWeight: 700, marginBottom: 10, fontFamily: "'JetBrains Mono', monospace" }}>
                              {phase.math.title}
                            </div>
                            {phase.math.equations.map((eq, i) => (
                              <div key={i} style={{ fontSize: 12, color: "#e2e8f0", fontFamily: "'JetBrains Mono', monospace", padding: "4px 0", lineHeight: 1.6, wordBreak: "break-all" }}>
                                {eq}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* Tasks */}
                      <div style={{ marginBottom: 16 }}>
                        <div style={{ fontSize: 11, color: "#64748b", fontWeight: 700, marginBottom: 8, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.08em" }}>TASKS</div>
                        <div style={{ display: "grid", gap: 4 }}>
                          {phase.tasks.map((t, i) => (
                            <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 12px", background: t.critical ? "#111827" : "transparent", borderRadius: 4, border: t.critical ? "1px solid #1e293b" : "1px solid transparent" }}>
                              <span style={{ fontSize: 10, color: t.critical ? "#f59e0b" : "#475569" }}>{t.critical ? "●" : "○"}</span>
                              <span style={{ flex: 1, fontSize: 12, color: t.critical ? "#e2e8f0" : "#94a3b8" }}>{t.task}</span>
                              <span style={{ fontSize: 11, color: "#64748b", fontFamily: "'JetBrains Mono', monospace", whiteSpace: "nowrap" }}>{t.effort}</span>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Deliverable & Risk */}
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                        <div style={{ background: "#0a0e1a", borderRadius: 6, padding: 14, border: `1px solid ${phase.color}33` }}>
                          <div style={{ fontSize: 10, color: phase.color, fontWeight: 700, marginBottom: 6, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.08em" }}>DELIVERABLE</div>
                          <div style={{ fontSize: 12, color: "#e2e8f0", lineHeight: 1.5 }}>{phase.deliverable}</div>
                        </div>
                        <div style={{ background: "#0a0e1a", borderRadius: 6, padding: 14, border: "1px solid #f43f5e33" }}>
                          <div style={{ fontSize: 10, color: "#f43f5e", fontWeight: 700, marginBottom: 6, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.08em" }}>RISK & MITIGATION</div>
                          <div style={{ fontSize: 12, color: "#fca5a5", lineHeight: 1.5 }}>{phase.risk}</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* DATA TAB */}
        {activeTab === "data" && (
          <div>
            <div style={{ marginBottom: 24, padding: 20, background: "#111827", borderRadius: 8, border: "1px solid #1e293b" }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, color: "#f59e0b", marginBottom: 8, fontFamily: "'JetBrains Mono', monospace" }}>
                ⚠️ 핵심 제약: 실험 불가 → 합성 + 오픈 데이터
              </h3>
              <p style={{ fontSize: 13, color: "#94a3b8", margin: 0, lineHeight: 1.6 }}>
                "회절 전용" 오픈 데이터셋은 존재하지 않는다. GWA의 FDTD 컴포넌트가 저주파 회절을 부분 포함하지만, 자네 연구에 필요한 "NLOS 회절 지배적 시나리오"는 직접 만들어야 한다. 3-Tier로 겹겹이 쌓는 전략이 필수.
              </p>
            </div>

            {DATA_TIERS.map((tier) => (
              <div key={tier.tier} style={{ marginBottom: 20, background: "#111827", borderRadius: 8, border: `1px solid ${tier.color}33`, overflow: "hidden" }}>
                <div style={{ padding: "14px 20px", borderBottom: "1px solid #1e293b", display: "flex", alignItems: "center", gap: 12 }}>
                  <Badge color={tier.color}>{tier.tier}</Badge>
                  <span style={{ fontSize: 15, fontWeight: 600 }}>{tier.name}</span>
                  <span style={{ fontSize: 12, color: "#64748b" }}>— {tier.purpose}</span>
                </div>
                <div style={{ padding: "12px 20px" }}>
                  {tier.datasets.map((ds, i) => (
                    <div key={i} style={{ padding: "12px 0", borderBottom: i < tier.datasets.length - 1 ? "1px solid #1e293b" : "none" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                        <span style={{ fontSize: 14, fontWeight: 700, color: tier.color }}>{ds.name}</span>
                        <span style={{ fontSize: 11, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>{ds.source}</span>
                        <span style={{ fontSize: 11, color: "#475569", fontFamily: "'JetBrains Mono', monospace" }}>{ds.size}</span>
                      </div>
                      <div style={{ fontSize: 12, color: "#22c55e", marginBottom: 4 }}>✓ {ds.value}</div>
                      {ds.warning !== "-" && <div style={{ fontSize: 12, color: "#f59e0b" }}>⚠ {ds.warning}</div>}
                    </div>
                  ))}
                </div>
              </div>
            ))}

            {/* Data Pipeline */}
            <div style={{ background: "#0a0e1a", border: "1px solid #1e293b", borderRadius: 8, padding: 20, marginTop: 24 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#a78bfa", marginBottom: 12, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.08em" }}>
                DATA PIPELINE FLOW
              </div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, lineHeight: 2.2, color: "#94a3b8" }}>
                <div><span style={{ color: "#f59e0b" }}>Tier 3</span> 해석해 → BEM 구현 검증 (오차 &lt; 3%)</div>
                <div style={{ color: "#475569" }}>{"  "}↓</div>
                <div><span style={{ color: "#8b5cf6" }}>Tier 2</span> BEM 합성 → 15장면 NLOS 회절 데이터 생성</div>
                <div style={{ color: "#475569" }}>{"  "}↓</div>
                <div><span style={{ color: "#3b82f6" }}>Tier 1</span> GWA/SoundSpaces → Pre-train + 일반화 테스트</div>
                <div style={{ color: "#475569" }}>{"  "}↓</div>
                <div><span style={{ color: "#10b981" }}>검증 </span> RAF 실측 → Sim-to-real gap 정량화</div>
              </div>
            </div>
          </div>
        )}

        {/* HARDWARE TAB */}
        {activeTab === "hardware" && (
          <div>
            <div style={{ display: "grid", gap: 12, marginBottom: 24 }}>
              {Object.entries(HARDWARE).map(([key, hw]) => (
                <div key={key} style={{ background: "#111827", borderRadius: 8, border: `1px solid ${hw.ok ? "#22c55e33" : "#f43f5e33"}`, padding: 20 }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <span style={{ fontSize: 18 }}>{key === "cpu" ? "🔲" : key === "ram" ? "📦" : "🎮"}</span>
                      <span style={{ fontSize: 16, fontWeight: 700 }}>{hw.name}</span>
                      <span style={{ fontSize: 12, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>{hw.spec}</span>
                    </div>
                    <Badge color={hw.ok ? "#22c55e" : "#f43f5e"}>{hw.ok ? "OK" : "BOTTLENECK"}</Badge>
                  </div>
                  <div style={{ fontSize: 13, color: hw.ok ? "#94a3b8" : "#fca5a5", lineHeight: 1.6 }}>{hw.limit}</div>
                </div>
              ))}
            </div>

            <div style={{ background: "#111827", borderRadius: 8, border: "1px solid #1e293b", padding: 20, marginBottom: 24 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#f59e0b", marginBottom: 14, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.08em" }}>
                RTX 2080 SUPER (8GB VRAM) 최적 설정
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                {[
                  { label: "SIREN 구조", value: "6층 × 512 뉴런 (~1.5M)", note: "8층 이상은 VRAM 초과" },
                  { label: "Fourier Features", value: "128 차원", note: "σ ≈ f_max/c ≈ 23 m⁻¹" },
                  { label: "Batch (collocation)", value: "2,048-4,096", note: "2차미분으로 메모리 2-3x" },
                  { label: "필수 최적화", value: "FP16 + Grad Checkpoint", note: "없으면 OOM 확정" },
                ].map((item, i) => (
                  <div key={i} style={{ background: "#0a0e1a", borderRadius: 6, padding: 14, border: "1px solid #1e293b" }}>
                    <div style={{ fontSize: 11, color: "#64748b", fontFamily: "'JetBrains Mono', monospace", marginBottom: 4 }}>{item.label}</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: "#e2e8f0" }}>{item.value}</div>
                    <div style={{ fontSize: 11, color: "#f59e0b", marginTop: 4 }}>{item.note}</div>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ background: "#111827", borderRadius: 8, border: "1px solid #1e293b", padding: 20 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#3b82f6", marginBottom: 14, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.08em" }}>
                BEM (CPU) 계산량 추정
              </div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, lineHeight: 2, color: "#94a3b8" }}>
                <div>메쉬 N=10,000 → 단일 freq BEM: <span style={{ color: "#22c55e" }}>1-3분</span></div>
                <div>15장면 × 75쌍 × 30freq = 33,750 solve</div>
                <div>예상 총 시간: <span style={{ color: "#f59e0b" }}>약 1-2주 (CPU 풀가동)</span></div>
                <div>메모리: 16·N² bytes → N=10K에서 <span style={{ color: "#22c55e" }}>~1.6GB (OK)</span></div>
                <div style={{ color: "#f43f5e" }}>⚠ N=50K 이상 → 40GB+ (불가)</div>
              </div>
            </div>
          </div>
        )}

        {/* RISK TAB */}
        {activeTab === "risk" && (
          <div>
            {[
              {
                level: "CRITICAL",
                color: "#f43f5e",
                items: [
                  {
                    title: "Venue Mismatch",
                    desc: "연구 내용은 Computational Acoustics인데, 전공은 Computer Vision. CVPR 리뷰어가 'ICASSP에 내야 하는 거 아닌가?'라고 물을 수 있다.",
                    solution: "Framing 전환: 'Seeing Around Corners with Sound' — NLOS imaging의 청각 버전으로 포지셔닝. Vision과의 연결고리(SDF → mesh → 시각적 렌더링)를 논문 첫 문장에 명시.",
                  },
                  {
                    title: "실험 데이터 부재",
                    desc: "100% 합성 데이터 → CVPR 추세상 real-world validation 없으면 불리.",
                    solution: "RAF(Google) 실측 데이터와 반드시 비교. 최소한 'sim과 real이 X dB 이내'를 보여야 한다. 향후 실측 계획을 Limitation에 명시.",
                  },
                ],
              },
              {
                level: "HIGH",
                color: "#f59e0b",
                items: [
                  {
                    title: "VRAM 8GB에서 Joint Learning 수렴 실패",
                    desc: "Forward(p) + SDF(s) + Helmholtz + Eikonal + BC를 동시에 학습하면 OOM 또는 수렴 불가.",
                    solution: "Alternating Training: Step 1) p만, Step 2) SDF만, Step 3) Joint. Gradient checkpointing + FP16 필수. 안 되면 SDF head를 별도 경량 네트워크로 분리.",
                  },
                  {
                    title: "BEM 계산 시간 초과",
                    desc: "33,750 BEM solve에 2주는 최적 시나리오. 메쉬 품질 이슈, 수렴 실패 등으로 지연 가능.",
                    solution: "주파수를 30 → 15로 축소 + 보간. 장면을 15 → 8로 축소. 백업: pyroomacoustics (ISM 기반, 빠르지만 회절 부정확).",
                  },
                  {
                    title: "PINN Spectral Bias",
                    desc: "고주파(k > 50 rad/m)에서 Helmholtz residual 수렴이 100배 느려짐.",
                    solution: "Fourier Features (σ ≈ f_max/c), Multi-scale curriculum (저주파→고주파), SIREN ω₀ ∝ k 스케일링. 최소 하나는 적용 필수.",
                  },
                ],
              },
              {
                level: "MEDIUM",
                color: "#3b82f6",
                items: [
                  {
                    title: "Bempp-cl Time-domain 미지원",
                    desc: "Bempp-cl은 주파수 영역 전용. Transient 시뮬레이션을 직접 지원하지 않음.",
                    solution: "Multi-freq solve + IDFT. 백업: k-Wave (MATLAB/Python) for time-domain. 또는 주파수 영역에서만 연구 진행 (논문 scope 조정).",
                  },
                  {
                    title: "GWA 회절 정확도 미검증",
                    desc: "GWA의 FDTD 부분이 2-8kHz에서 회절을 얼마나 정확히 포착하는지 불확실.",
                    solution: "Phase 1에서 BEM과 GWA를 동일 장면에서 비교 → 신뢰 주파수 대역 확인 후 사용.",
                  },
                  {
                    title: "Contribution 산만",
                    desc: "Structured Green + SDF + PINN + Cycle = 4개의 작은 contribution → 날카로움 부족.",
                    solution: "'단일 마이크로 NLOS 기하구조 최초 복원'이라는 하나의 화살로 수렴. 나머지는 이를 위한 도구.",
                  },
                ],
              },
            ].map((group) => (
              <div key={group.level} style={{ marginBottom: 20 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                  <Badge color={group.color}>{group.level}</Badge>
                </div>
                {group.items.map((item, i) => (
                  <div key={i} style={{ background: "#111827", border: "1px solid #1e293b", borderRadius: 8, padding: 16, marginBottom: 8, borderLeft: `3px solid ${group.color}` }}>
                    <div style={{ fontSize: 14, fontWeight: 700, color: "#e2e8f0", marginBottom: 6 }}>{item.title}</div>
                    <div style={{ fontSize: 12, color: "#f87171", marginBottom: 8, lineHeight: 1.5 }}>{item.desc}</div>
                    <div style={{ fontSize: 12, color: "#22c55e", lineHeight: 1.5, padding: "8px 12px", background: "#22c55e0a", borderRadius: 4, border: "1px solid #22c55e22" }}>
                      → {item.solution}
                    </div>
                  </div>
                ))}
              </div>
            ))}

            {/* Self-Critique */}
            <div style={{ background: "linear-gradient(135deg, #1e1b4b, #172554)", border: "1px solid #a78bfa44", borderRadius: 8, padding: 20, marginTop: 24 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#a78bfa", marginBottom: 12, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.08em" }}>
                🔬 자기 비판 (SELF-CRITIQUE ON THIS ROADMAP)
              </div>
              {[
                "과신 편향: v3.2가 '완벽하다'고 했지만 실제 구현에서 예상치 못한 문제가 반드시 나온다. BEM 수치 불안정성, Loss 밸런싱의 어려움.",
                "복잡성 편향: 모든 요소를 한꺼번에 구현하면 디버깅이 악몽. 반드시 점진적 통합(Incremental)으로 진행할 것.",
                "타임라인 낙관: 18개월에 CVPR은 '모든 것이 순조로울 때'의 시나리오. 현실적 마음속 목표는 24개월.",
                "2D/2.5D 한계: 32GB RAM으로는 복잡한 3D BEM 불가. 이건 논문 Limitation에 솔직히 쓰는 게 맞다.",
              ].map((item, i) => (
                <div key={i} style={{ fontSize: 12, color: "#c4b5fd", lineHeight: 1.6, padding: "4px 0 4px 14px", borderLeft: "2px solid #a78bfa44", marginBottom: 6 }}>
                  {item}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
