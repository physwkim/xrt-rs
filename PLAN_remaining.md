# 남은 2% 구현 계획

현재 API 커버리지 ~98%. 남은 항목은 **새 기능 구현** 또는 **고도의 물리 분석**이 필요.

---

## 1. CRL (Compound Refractive Lens) 파이프라인

### 현재 상태
- `ParaboloidLensSurface` 완전 구현됨 (`surfaces/lens.rs`, 100 LOC, 4 unit tests)
- **하지만 OE 래퍼가 없어서** Beamline에 통합 불가
- DeflectionMode::Refract은 이미 작동 확인 (beamline E2E 통과)

### 구현 계획

```
Phase 1: Single lens OE (1일)
├── xrt-oes/src/lens_oe.rs — LensOpticalElement 생성
│   ├── ParaboloidLensSurface + DeflectionMode::Refract 조합
│   ├── material로부터 n1_over_n2 자동 계산 (에너지 의존)
│   └── Beamline::add_lens() 메서드 추가
└── 테스트: Source → Lens → Screen, 초점 검증

Phase 2: CRL 스택 (1일)
├── N개 렌즈 자동 배치 (간격 = thickness)
├── Beamline::add_crl(n_lenses, lens_params) 편의 메서드
└── 테스트: Source → CRL(10) → Screen, 1/f_total = N/f_single 검증

Phase 3: Python 바인딩 (0.5일)
├── crl_focus_rs(n_lenses, R, material, energy) → focal_length
└── pytest: Rust vs 해석적 초점거리 비교
```

### 핵심 수식
```
1/f = δ/R  (single lens)
1/f_total = N/f_single  (CRL)
δ = (r_e × λ² × ρ × N_A) / (2π × A)  (refractive decrement)
```

### 난이도: ★★☆ (기존 코드 조합, 새 물리 없음)

---

## 2. Mosaicity 구현 — ✅ 구현 완료

### 현재 상태
- `CrystalBase.mosaicity: f64` 필드 저장 (line 107)
- `get_darwin_width()`에서 `width_total = sqrt(dynamical² + mosaicity²)` 보정 적용 (crystal.rs:230)
- `get_amplitude()`에서 mosaicity > 0일 때 Gaussian convolution 경로 활성화 (crystal.rs:243)
  - 11-point weighted sum over angular offsets ±3σ

### 남은 작업
- XRT Python과의 cross-validation 테스트 추가 (mosaicity > 0 조건)

### 핵심 물리
```
R_mosaic(θ) = ∫ R_perfect(θ-Δθ) × G(Δθ, σ_m) dΔθ
σ_m = mosaicity (라디안)
```

### 난이도: ★★★ (물리 구현 + XRT 컨볼루션 방식 역공학)

---

## 3. Undulator 고조파 스펙트럼 피크 검출

### 현재 상태
- `build_i_map(energies, thetas, psis)` 가 trajectory 적분으로 스펙트럼 계산
- 고조파 피크는 **자연스럽게 나타남** (N-period coherence)
- `fundamental_energy()` = E₁ ≈ 8044 eV (6GeV, K=1.5, λ=20mm)

### 구현 계획

```
Phase 1: 스펙트럼 스캔 테스트 (0.5일)
├── E₁ 주변 fine scan: build_i_map([E₁×0.5 .. E₁×5.5], [0], [0])
├── 피크 검출: local maxima at E ≈ n×E₁ (n=1,3,5)
└── 테스트:
    assert peak_energies ≈ [E₁, 3E₁, 5E₁] (±10%)
    assert I(E₃) < I(E₁)  (고조파 감소)

Phase 2: On-axis vs off-axis (0.5일)
├── theta=0: 홀수 고조파만 (n=1,3,5)
├── theta>0: 짝수 고조파도 나타남 (n=2,4)
└── 테스트:
    on_axis: I(2E₁) ≈ 0
    off_axis: I(2E₁) > 0
```

### 핵심 물리
```
E_n = n × E₁ (on-axis, odd n only)
I_n ∝ K²n² × [J_{(n-1)/2}(nK²/(4+2K²)) - J_{(n+1)/2}(...)]²
```

### 난이도: ★★☆ (기존 API 사용, 분석만 추가)

---

## 4. Wiggler K→0 극한

### 현재 상태
- K→0에서 코드는 `cos_arg = θγ/K → ∞` → `w_cr = 0`
- 물리적으로 K→0 wiggler는 BM과 다름 (wiggler는 주기적, BM은 단일 자석)
- **K→0에서 BM 수렴은 물리적으로 부정확** — wiggler는 N_periods × BM flux

### 구현 계획

```
Phase 1: 문서화 (즉시)
└── wiggler.rs에 docstring 추가:
    "K→0 limit: wiggler becomes highly collimated,
     does NOT converge to single BM.
     For BM-equivalent, use BendingMagnet directly."

Phase 2: Flux scaling 테스트 (0.5일)
├── 고정 K에서 N_periods 변화: flux ∝ N_periods
├── 테스트: wiggler(N=20) flux ≈ 2 × wiggler(N=10) flux
└── 이미 유사 테스트 존재 (wiggler_flux_higher_than_bm)

Phase 3: Critical energy 일치 테스트 (0.5일)
├── Wiggler Ec = 0.665 × E²(GeV) × B_max(T)
│   where B_max = K / (0.0934 × period_cm)
└── 테스트: 스펙트럼 median energy ≈ 0.3 × Ec (universal curve)
```

### 난이도: ★☆☆ (주로 문서화 + 간단한 스케일링 테스트)

---

## 5. Phase Wrapping 정밀도

### 현재 상태
- CPU: f64 `sin_cos(phase)` — phase가 아무리 커도 정확 (f64는 2^53 ≈ 9e15까지 정확)
- GPU: algebraic phase reduction으로 이미 해결됨 (0.01% 오차)
- 실질적으로 **문제 없음**

### 구현 계획

```
Phase 1: 극한 테스트 추가 (0.5일)
├── CPU: phase = 2π × 10⁹ (10억 바퀴) → sin(phase) 정확도
├── GPU: 극장거리 (path=10000mm) 에서 phase reduction 검증
└── 테스트:
    CPU sin(2π×N + ε) ≈ sin(ε) within 1e-10
    GPU long-distance: nonzero output

Phase 2: 문서화 (즉시)
└── diffraction.rs docstring:
    "CPU f64: precise to ~1e-15 for phase up to ~1e15 rad
     GPU f32: algebraic reduction keeps delta_phase < 1e6 rad"
```

### 난이도: ★☆☆ (이미 거의 해결됨)

---

## 6. Grating 효율 vs 에너지

### 현재 상태
- `grating_deflection()` 은 **방향만** 계산, 효율 없음
- Material amplitude는 `GratingOpticalElement` 에서 적용 (표면 반사율)
- **Blaze 효율 (order-dependent)** 은 전혀 구현 안 됨

### 구현 계획

```
Phase 1: Blaze 효율 함수 (1일)
├── deflection.rs에 blaze_efficiency() 추가:
│   η(λ, m, α, γ) = sinc²(π × m × cos(α)/cos(γ) - π × d × (sin(α)+sin(γ))/λ)
│   where α=blaze angle, γ=diffracted angle, d=groove spacing
├── BlazedGrating에 blaze_efficiency(energy, order, theta_in, theta_out) 메서드
└── 단위 테스트: order=1이 blaze angle 근처에서 최대 효율

Phase 2: GratingOpticalElement 통합 (1일)
├── reflect.rs에서 grating deflection 후 blaze efficiency factor 적용
├── amplitude *= efficiency_factor per ray
└── 테스트: 에너지 스캔 → efficiency curve가 blaze peak 보여야 함

Phase 3: Python XRT cross-comparison (0.5일)
├── XRT의 grating efficiency 계산과 비교
└── 에너지 범위 500-2000 eV, order ±1 ±2
```

### 핵심 수식
```
η_blaze(m, λ) = sinc²(mπ - πd(sinα + sinγ)/λ)
Peak efficiency at: mλ = d(sinα + sinγ_blaze)
```

### 난이도: ★★★ (새 물리 구현 + 기존 파이프라인 수정)

---

## 7. Criterion 벤치마크 확장

### 현재 상태
- 3개 벤치마크 스위트 존재:
  - `xrt-core/benches/transforms.rs` — rotate_beam, rotate_xyz
  - `xrt-oes/benches/intersection.rs` — intersection, parallel_reflect
  - `xrt-sources/benches/sources.rs` — source generation

### 구현 계획

```
Phase 1: 누락 벤치마크 추가 (1일)
├── xrt-waves/benches/diffraction.rs
│   ├── CPU Kirchhoff: 100 rays × 100 pixels → 10K × 10K
│   └── GPU Kirchhoff: 동일 사이즈, GPU throughput 측정
├── xrt-materials/benches/crystal.rs
│   ├── get_bragg_angle: 1K ~ 1M 에너지 포인트
│   ├── get_amplitude: 1K ~ 100K 각도 포인트
│   └── get_f_chi: 1K ~ 100K
└── xrt-pytte/benches/tt_solver.rs
    └── solve_bragg_parallel: 100 ~ 10K 포인트

Phase 2: 회귀 방지 (0.5일)
├── CI에 `cargo bench` 통합 (GitHub Actions)
├── 결과를 JSON으로 저장 → 이전 결과와 비교
└── >10% 성능 저하 시 warning

Phase 3: GPU vs CPU 비교 보고서 (0.5일)
├── 동일 입력에 대해 GPU와 CPU 시간 비교
├── Crossover point: N rays × N pixels에서 GPU가 유리해지는 크기
└── README에 성능 테이블 추가
```

### 난이도: ★★☆ (코드 작성은 간단, CI 통합이 핵심)

---

## 우선순위 및 일정

| # | 항목 | 난이도 | 예상 시간 | 의존성 |
|---|------|--------|----------|--------|
| 1 | Phase wrapping 테스트 | ★☆☆ | 0.5일 | 없음 |
| 2 | Wiggler K→0 문서화 + flux scaling | ★☆☆ | 0.5일 | 없음 |
| 3 | Undulator 고조파 피크 검출 | ★★☆ | 1일 | 없음 |
| 4 | CRL 파이프라인 | ★★☆ | 2.5일 | Refract 작동 확인 ✅ |
| 5 | Criterion 벤치마크 | ★★☆ | 2일 | 없음 |
| 6 | Mosaicity 구현 | ★★★ | 2일 | Crystal amplitude ✅ |
| 7 | Grating 효율 | ★★★ | 2.5일 | Grating deflection ✅ |

**총 예상: ~11일**

```
Week 1: #1 Phase wrapping → #2 Wiggler → #3 Undulator harmonics
Week 2: #4 CRL pipeline (3 phases)
Week 3: #5 Benchmarks → #6 Mosaicity
Week 4: #7 Grating efficiency
```
