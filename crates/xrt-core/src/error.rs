use thiserror::Error;

/// Errors that can occur during X-ray tracing operations.
#[derive(Debug, Error)]
#[allow(dead_code)] // Some variants reserved for future use
pub enum XrtError {
    #[error("beam size mismatch: expected {expected}, got {got}")]
    BeamSizeMismatch { expected: usize, got: usize },

    #[error("invalid rotation sequence: {0}")]
    InvalidRotationSequence(String),

    #[error("invalid ray state: {0}")]
    InvalidRayState(i32),

    #[error("index out of bounds: index {index}, length {length}")]
    IndexOutOfBounds { index: usize, length: usize },

    #[error("energy {energy} eV outside table range [{min}, {max}]")]
    EnergyOutOfRange { energy: f64, min: f64, max: f64 },

    #[error("element not found: {0}")]
    ElementNotFound(String),

    #[error("data file parse error: {0}")]
    DataParse(String),

    #[error("root finding failed after {iterations} iterations")]
    RootFindConvergence { iterations: usize },

    #[error("invalid crystal geometry: {0}")]
    InvalidCrystalGeometry(String),

    #[error("interpolation error: {0}")]
    Interpolation(String),

    #[error("shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, XrtError>;
