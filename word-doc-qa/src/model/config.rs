/// Hyperparameter configuration for the QA transformer model.

use serde::{Deserialize, Serialize};

/// All architectural hyperparameters in one serialisable struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Vocabulary size (must match the tokenizer).
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Model (embedding) dimension d_model.
    pub d_model: usize,
    /// Number of attention heads (d_model must be divisible by n_heads).
    pub n_heads: usize,
    /// Feed-forward hidden dimension inside each encoder layer.
    pub d_ff: usize,
    /// Number of transformer encoder layers (minimum 6 per spec).
    pub n_layers: usize,
    /// Dropout probability applied inside encoder layers.
    pub dropout: f64,
}

impl ModelConfig {
    /// Default "small" configuration — fast to train on a laptop.
    pub fn small(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            max_seq_len: 256,
            d_model: 128,
            n_heads: 4,
            d_ff: 256,
            n_layers: 6,
            dropout: 0.1,
        }
    }

    /// Default "medium" configuration — better capacity, needs more VRAM.
    pub fn medium(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            max_seq_len: 256,
            d_model: 256,
            n_heads: 8,
            d_ff: 512,
            n_layers: 6,
            dropout: 0.1,
        }
    }

    /// Validate configuration consistency.
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model % self.n_heads != 0 {
            return Err(format!(
                "d_model ({}) must be divisible by n_heads ({})",
                self.d_model, self.n_heads
            ));
        }
        if self.n_layers < 6 {
            return Err(format!(
                "n_layers ({}) must be at least 6 per assignment spec",
                self.n_layers
            ));
        }
        Ok(())
    }
}
