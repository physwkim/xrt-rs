// Undulator radiation compute shader — WGSL
//
// Computes undulator radiation amplitudes by numerically integrating
// the electron trajectory over one period.
// Each workgroup processes one observation point (energy, theta, psi).

struct UndulatorParams {
    n_points: u32,
    n_steps: u32,        // integration steps per period
    n_periods: u32,
    _pad0: u32,
    kx: f32,
    ky: f32,
    gamma: f32,
    gamma2: f32,
    period_m: f32,       // period in meters
    phase_rad: f32,      // Kx-Ky phase difference [rad]
    amp2flux: f32,
    _pad1: f32,
};

struct ObsPoint {
    energy: f32,         // [eV]
    theta: f32,          // [rad]
    psi: f32,            // [rad]
    _pad: f32,
};

struct PointResult {
    ax_re: f32,
    ax_im: f32,
    az_re: f32,
    az_im: f32,
    intensity: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> obs_points: array<ObsPoint>;
@group(0) @binding(1) var<storage, read_write> results: array<PointResult>;
@group(0) @binding(2) var<uniform> params: UndulatorParams;

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const E2W: f32 = 1519267514747457.9;  // eV → rad/s
const SIC: f32 = 299792458.0;         // speed of light [m/s]

/// sin(x)/x, from its series where the quotient is ill-conditioned.
///
/// A backend's sin() is only accurate in absolute terms, so for an argument of
/// 1e-6 its answer can be several percent off in relative terms - and the
/// N-period resonance asks for exactly that ratio of two vanishing sines at
/// every harmonic. Taking it as a ratio of sincs keeps both operands near 1.
fn sinc(x: f32) -> f32 {
    if (abs(x) < 0.1) {
        let x2 = x * x;
        return 1.0 - x2 / 6.0 * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0));
    }
    return sin(x) / x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_points) {
        return;
    }

    let obs = obs_points[idx];
    let omega = obs.energy * E2W;
    let dt = 1.0 / f32(params.n_steps);

    var ax_re: f32 = 0.0;
    var ax_im: f32 = 0.0;
    var az_re: f32 = 0.0;
    var az_im: f32 = 0.0;

    let k2 = params.kx * params.kx + params.ky * params.ky;

    // As in the CPU Undulator::build_i_map: q = 1 - β̄ is how far the electron
    // falls behind light per unit path, `slip` is 1 - dirz·β̄ assembled from its
    // two small parts (forming it by subtraction would cancel away every f32
    // digit), and `wiggle_amp` is the amplitude of the K²-driven longitudinal
    // wiggle that puts the harmonics on axis.
    let q = (1.0 + 0.5 * k2) / (2.0 * params.gamma2);
    let off_axis = obs.theta * obs.theta + obs.psi * obs.psi;
    let dirz = sqrt(1.0 - off_axis);
    let slip = off_axis / (1.0 + dirz) + q * dirz;
    let wc = omega / (SIC * (1.0 - q));
    let wiggle_amp = params.period_m / (8.0 * TWO_PI * params.gamma2);

    for (var j: u32 = 0u; j < params.n_steps; j = j + 1u) {
        let t = (f32(j) + 0.5) * dt;
        let phi_t = TWO_PI * t;

        // Electron velocity and position, as in the CPU
        // Undulator::build_i_map: K_y (the vertical field) bends the electron
        // horizontally, K_x vertically, and the minus sign on the K_x term
        // carries Python's sense of rotation (sources/synchr.py:48-51).
        let beta_x = params.ky / params.gamma * sin(phi_t);
        let beta_z = -params.kx / params.gamma * sin(phi_t + params.phase_rad);

        // Electron position
        let amp_x = params.ky * params.period_m / (TWO_PI * params.gamma);
        let amp_z = params.kx * params.period_m / (TWO_PI * params.gamma);
        let x_e = -amp_x * cos(phi_t);
        let z_e = amp_z * cos(phi_t + params.phase_rad);

        // Retarded phase, as in the CPU: s - dirz·(β̄s + wiggle) expanded into
        // s·slip - dirz·wiggle, which advances 2π per period at E₁.
        let s = params.period_m * t;
        let wiggle = wiggle_amp * (params.ky * params.ky * sin(2.0 * phi_t)
            + params.kx * params.kx * sin(2.0 * phi_t + 2.0 * params.phase_rad));
        let total_phase = wc * (s * slip - dirz * wiggle - x_e * obs.theta - z_e * obs.psi);

        let cos_p = cos(total_phase);
        let sin_p = sin(total_phase);

        // Accumulate radiation amplitude
        let bx_eff = beta_x - obs.theta;
        let bz_eff = beta_z - obs.psi;

        ax_re += bx_eff * cos_p * dt;
        ax_im += bx_eff * sin_p * dt;
        az_re += bz_eff * cos_p * dt;
        az_im += bz_eff * sin_p * dt;
    }

    // The N periods differ only by the phase the electron slips in one of
    // them, so the device amplitude is the single-period integral times
    // Σ_{n<N} exp(inΔφ): N at every harmonic, near zero between them. Folded
    // onto ±π/2 as in the CPU period_sum, which keeps the identity exact and
    // the sines out of the rounding noise.
    let n_per = f32(params.n_periods);
    let half = 0.5 * wc * params.period_m * slip;
    let d = half - PI * round(half / PI);
    let mag = n_per * sinc(n_per * d) / sinc(d);
    let arg = (n_per - 1.0) * d;
    let res_re = cos(arg) * mag;
    let res_im = sin(arg) * mag;

    let ax_res_re = ax_re * res_re - ax_im * res_im;
    let ax_res_im = ax_re * res_im + ax_im * res_re;
    let az_res_re = az_re * res_re - az_im * res_im;
    let az_res_im = az_re * res_im + az_im * res_re;

    ax_re = ax_res_re * params.gamma2;
    ax_im = ax_res_im * params.gamma2;
    az_re = az_res_re * params.gamma2;
    az_im = az_res_im * params.gamma2;

    // Intensity = |ax|² + |az|²
    let is_val = ax_re * ax_re + ax_im * ax_im;
    let ip_val = az_re * az_re + az_im * az_im;
    let inv_e = 1.0 / obs.energy;
    let total_intensity = params.amp2flux * inv_e * (is_val + ip_val);

    // Scale amplitudes
    let sqrt_flux = sqrt(params.amp2flux * inv_e);
    results[idx].ax_re = ax_re * sqrt_flux;
    results[idx].ax_im = ax_im * sqrt_flux;
    results[idx].az_re = az_re * sqrt_flux;
    results[idx].az_im = az_im * sqrt_flux;
    results[idx].intensity = total_intensity;
    results[idx]._pad0 = 0.0;
    results[idx]._pad1 = 0.0;
    results[idx]._pad2 = 0.0;
}
