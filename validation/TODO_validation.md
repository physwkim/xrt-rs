# Validation Framework — 추가 가능 항목

현재: Rust golden **63+**, pytest **76** pass. 공개 API 커버리지 ~82%.

---

## Priority 1: CRITICAL (미검증 핵심 기능)

### ~~1.1 GPU Kirchhoff 검증~~ ✅ DONE
- [x] GPU vs CPU 수치 동등성 (0.01% 오차, tolerance 1e-3)
- [x] Algebraic phase reduction으로 장거리(1000mm) 정밀도 확보
- [x] GPU symmetry, auto dispatch, non-zero output 검증
- [x] f32 precision 버그 발견 및 수정 (README에 기술)

### ~~1.2 Crystal Laue/Transmitted geometry~~ ✅ DONE
- [x] BraggTransmitted, LaueReflected, LaueTransmitted 진폭 계산
- [x] Python XRT cross-comparison (Bragg transmitted)
- [x] thickness 의존성 (Pendellösung) 테스트
- [ ] Laue geometry rocking curve scan (multi-angle)

### ~~1.3 Parametric surface intersection~~ ✅ DONE
- [x] elliptical, parabolical, hyperbolic round-trip pytest
- [x] 대칭성 및 on-axis 검증
- [ ] Python XRT OE 클래스 cross-comparison (수치 비교)

### ~~1.4 Multilayer roughness~~ ✅ PARTIAL
- [x] Roughness σ > 0 Nevot-Croce 단조감소 검증
- [x] Zero-roughness Rust↔Python cross-comparison
- [ ] `MultilayerGeom::Transmitted` 계산 검증
- [ ] 단일 bilayer (n_pairs=1) 경계조건

---

## Priority 2: HIGH (물리적 완전성)

### ~~2.1 Beamline 다중 소자 파이프라인~~ ✅ PARTIAL
- [x] `Beamline::add_grating()` 통합 테스트
- [x] Multi-element pipeline (2 mirrors + drift)
- [ ] `Beamline::add_crystal()` 통합 테스트
- [ ] 3+ 소자 체이닝 (Source → Mirror → Crystal → Screen)
- [ ] Coherency matrix (jss, jpp, jsp) 파이프라인 보존 검증

### ~~2.2 Grating deflection 확장~~ ✅ PARTIAL
- [x] 음의 회절차수 (m = -1)
- [ ] Evanescent wave 조건 (u < 0)
- [ ] Explicit sign override (`sig=Some(value)`)
- [ ] Grating 효율 vs 에너지 스캔
- **Why:** 양의 차수만 테스트됨. 연X선 분광기는 음의 차수 사용

### 2.3 Refraction (Snell) 확장
- [ ] `DeflectionMode::Refract { n1_over_n2 }` 통합 테스트
- [ ] 전반사 (total internal reflection) 임계각
- [ ] n1 > n2 (dense → sparse) 경로
- [ ] CRL (compound refractive lens) 파이프라인
- **Why:** 렌즈 기반 광학 미검증

### ~~2.4 Crystal 물성 확장~~ ✅ PARTIAL
- [x] Debye-Waller factor 변화 (chih 감소 검증)
- [x] `CrystalDiamond` variant (Ge(111)) Bragg angle
- [x] Si(333), Si(444) 고차 반사 Bragg angle
- [ ] Mosaicity > 0 효과
- [ ] `CrystalFromCell` (arbitrary unit cell)

### ~~2.5 Undulator 심화~~ ✅ PARTIAL
- [x] K_x only (수직 편광 undulator)
- [x] K_x = K_y (equal deflection)
- [x] Near-zero K 엣지케이스
- [ ] `with_phase()` elliptical (PyO3 바인딩에 미노출)
- [ ] 고조파 에너지: n=3, n=5

---

## Priority 3: MEDIUM (엣지케이스 및 정밀도)

### ~~3.1 Geometric source 확장~~ ✅ PARTIAL
- [x] `SpatialDist::Annulus` (환형 빔) — bounds check
- [ ] Polarization: Plus45, Minus45, Left circular
- [ ] Rotation angles (pitch, roll, yaw)
- [ ] `EnergyDist::Lines` weighted selection

### ~~3.2 Synchrotron source 정밀 검증~~ ✅ PARTIAL
- [x] BM `from_rho()` 대체 생성자 — direction normalization
- [ ] Position sampling (dx, dz > 0) 공간 분포
- [ ] Polarization 출력 (jss, jpp, jsp) 검증
- [ ] Wiggler K→0 극한 (BM과 동등)
- [ ] BM 임계에너지 스펙트럼 검증 (synchrotron universal curve)

### 3.3 Diffraction 정밀도
- [ ] Ray at pixel location (path → 0) 예외처리
- [ ] Polychromatic 빔 (에너지 분산 있는 rays)
- [ ] 대규모 ray set (N > 10000) Kahan summation 안정성
- [ ] Phase wrapping 2πn 정밀도

### ~~3.4 Surface 엣지케이스~~ ✅ PARTIAL
- [x] Toroid: x ≈ r (sagittal 경계) — finite check
- [x] Spherical: ρ ≈ R 및 ρ > R (clamp 검증)
- [x] Blazed grating: non-boundary y 값 (z > 0, unit normal)
- [ ] LaminarGrating `local_g()` 검증
- [ ] FZP zone boundary 전이

### 3.5 Material 엣지케이스
- [ ] `Multilayer::period()` 값 검증
- [ ] Material 밀도 ρ=0 예외처리
- [ ] 복합 compound: 3+ 원소 (예: LaAlO3)
- [ ] 극저에너지 (< 100 eV) 영역 scattering

---

## Priority 4: LOW (인프라 및 품질)

### 4.1 DeflectionMode::PassThrough
- [ ] 투과 모드에서 beam 좌표 변환 없이 통과
- [ ] 투과 crystal slab에서 흡수만 적용

### 4.2 Property-based testing 확장
- [ ] 에너지 보존: 반사율² + 투과율² ≤ 1 (proptest)
- [ ] 방향코사인 단위벡터 불변성 (모든 변환 후)
- [ ] 대칭 광학계 → 대칭 빔 (proptest)
- [ ] Bragg angle: dθ → 0이면 |Rs| → 1 (proptest)

### 4.3 Error path 테스트
- [ ] 잘못된 원소 이름 → Error 반환
- [ ] 음의 에너지 → Error
- [ ] 비물리적 파라미터 (R < 0, rho < 0)
- [ ] PyO3 타입 오류 처리

### 4.4 벤치마크 검증
- [ ] Criterion 벤치마크 결과 회귀 방지
- [ ] GPU vs CPU throughput 비교
- [ ] Rayon parallelism 스케일링

---

## 수치 요약

| 카테고리 | 전체 공개 API | 테스트됨 | 미테스트 | 커버리지 |
|----------|-------------|---------|---------|---------|
| xrt-sources | 18 | 12 | 6 | 67% |
| xrt-oes | 25 | 15 | 10 | 60% |
| xrt-materials | 15 | 10 | 5 | 67% |
| xrt-waves | 4 | 2 | 2 | 50% |
| xrt-gpu | 2 | 0 | 2 | 0% |
| xrt-python PyO3 | 10 | 8 | 2 | 80% |
| **합계** | **74** | **47** | **27** | **64%** |

---

## 실행 순서 권장

```
1.2 Crystal Laue → 1.1 GPU → 1.3 Parametric → 1.4 Multilayer transmitted
→ 2.1 Beamline multi → 2.2 Grating → 2.5 Undulator → 2.3 Refraction
→ 2.4 Crystal variants → 3.x Edge cases → 4.x Infra
```
