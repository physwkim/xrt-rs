# Validation Framework — 추가 가능 항목

현재: Rust golden 45, pytest 62 pass. 공개 API 커버리지 ~64%.

---

## Priority 1: CRITICAL (미검증 핵심 기능)

### 1.1 GPU Kirchhoff 검증 (커버리지: 0%)
- [ ] GPU vs CPU 수치 동등성 테스트 (`kirchhoff_gpu()` vs `diffraction_integral()`)
- [ ] GPU 버퍼 전송 정확성
- [ ] GPU fallback 경로 검증
- **Why:** GPU 경로가 전혀 검증되지 않음. production에서 GPU 사용 시 무검증 상태

### 1.2 Crystal Laue/Transmitted geometry (커버리지: 0%)
- [ ] `CrystalGeometry::LaueReflected` 진폭 계산
- [ ] `CrystalGeometry::LaueTransmitted` 진폭 계산
- [ ] `CrystalGeometry::BraggTransmitted` 진폭 계산
- [ ] Fixture 생성: Python XRT로 각 geometry에 대해 진폭 참조값
- [ ] thickness 유한값 (예: 0.1mm) 설정 시 Laue case 반사율 변화
- **Why:** 4개 geometry 중 1개만 테스트됨. Laue optics는 고에너지 X선 실험에 필수

### 1.3 Parametric surface intersection (커버리지: 0%)
- [ ] `find_intersection_parametric_rs()` for elliptical mirror
- [ ] `find_intersection_parametric_rs()` for parabolical mirror
- [ ] `find_intersection_parametric_rs()` for hyperbolic mirror
- [ ] Fixture: Python XRT parametric OE 클래스로 교차점 생성
- **Why:** PyO3 바인딩 존재하지만 pytest 전무

### 1.4 Multilayer transmitted geometry
- [ ] `MultilayerGeom::Transmitted` 계산 검증
- [ ] Python XRT `ml.get_amplitude()` transmitted 모드 참조값
- [ ] Roughness σ > 0 효과: Nevot-Croce factor 검증
- [ ] 단일 bilayer (n_pairs=1) 경계조건
- **Why:** 구현됨 but 반사만 테스트. X선 투과 광학 미검증

---

## Priority 2: HIGH (물리적 완전성)

### 2.1 Beamline 다중 소자 파이프라인
- [ ] `Beamline::add_grating()` 통합 테스트
- [ ] `Beamline::add_crystal()` 통합 테스트
- [ ] 3+ 소자 체이닝 (Source → Mirror → Crystal → Screen)
- [ ] Coherency matrix (jss, jpp, jsp) 파이프라인 보존 검증
- [ ] 편광도 (degree of polarization) 전파
- **Why:** 현재 Mirror → Screen만 E2E 테스트. 실제 빔라인은 복합 구성

### 2.2 Grating deflection 확장
- [ ] 음의 회절차수 (m = -1, -2)
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

### 2.4 Crystal 물성 확장
- [ ] Mosaicity > 0 효과
- [ ] Debye-Waller factor 변화 (fact_dw ≠ 1.0)
- [ ] `CrystalDiamond` variant (Ge, C)
- [ ] `CrystalFromCell` (arbitrary unit cell)
- [ ] Si(333), Si(444) 고차 반사
- **Why:** 현재 Si(111), Si(220)만 297.15K에서 테스트

### 2.5 Undulator 심화
- [ ] `with_phase()` elliptical polarization (phase 0°→180°)
- [ ] K_x only (수직 편광 undulator)
- [ ] K_x = K_y, phase=90° (원형 편광)
- [ ] 고조파 에너지: n=3, n=5
- [ ] Angular distribution 1/γ 검증
- **Why:** planar K_y only 테스트됨. 편광 제어 undulator 미검증

---

## Priority 3: MEDIUM (엣지케이스 및 정밀도)

### 3.1 Geometric source 확장
- [ ] `SpatialDist::Annulus` (환형 빔)
- [ ] Polarization: Plus45, Minus45, Left circular
- [ ] Rotation angles (pitch, roll, yaw)
- [ ] `EnergyDist::Normal(mean, sigma)`
- [ ] `EnergyDist::Lines` weighted selection

### 3.2 Synchrotron source 정밀 검증
- [ ] BM `from_rho()` 대체 생성자
- [ ] Position sampling (dx, dz > 0) 공간 분포
- [ ] Polarization 출력 (jss, jpp, jsp) 검증
- [ ] Wiggler K→0 극한 (BM과 동등)
- [ ] BM 임계에너지 스펙트럼 검증 (synchrotron universal curve)

### 3.3 Diffraction 정밀도
- [ ] Ray at pixel location (path → 0) 예외처리
- [ ] Polychromatic 빔 (에너지 분산 있는 rays)
- [ ] 대규모 ray set (N > 10000) Kahan summation 안정성
- [ ] Phase wrapping 2πn 정밀도

### 3.4 Surface 엣지케이스
- [ ] Toroid: x ≈ r (sagittal 경계)
- [ ] Spherical: ρ ≈ R (유효 영역 경계)
- [ ] Blazed grating: non-boundary y 값 (현재 전부 boundary)
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
