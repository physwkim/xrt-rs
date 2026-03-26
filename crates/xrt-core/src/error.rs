use thiserror::Error;

#[derive(Debug, Error)]
pub enum XrtError {
    #[error("beam size mismatch: expected {expected}, got {got}")]
    BeamSizeMismatch { expected: usize, got: usize },

    #[error("invalid rotation sequence: {0}")]
    InvalidRotationSequence(String),

    #[error("invalid ray state: {0}")]
    InvalidRayState(i32),

    #[error("index out of bounds: index {index}, length {length}")]
    IndexOutOfBounds { index: usize, length: usize },

    #[error("shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, XrtError>;
