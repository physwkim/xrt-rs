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
        let y_e = params.period_m * t;

        // Phase
        let path = y_e - x_e * obs.theta - z_e * obs.psi;
        let phase_term = omega / SIC * path;
        let avg_correction = omega / SIC * params.period_m * t * (1.0 + k2 / 2.0) / (2.0 * params.gamma2);
        let total_phase = phase_term - avg_correction;

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

    // Scale by N_periods and gamma²
    let n_per = f32(params.n_periods);
    let scale = n_per * params.gamma2;

    ax_re *= scale;
    ax_im *= scale;
    az_re *= scale;
    az_im *= scale;

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
