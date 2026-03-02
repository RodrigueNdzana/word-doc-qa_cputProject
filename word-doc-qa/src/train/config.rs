#![allow(dead_code)]
/// Training hyperparameter configuration (serialisable for CLI config files).

use serde::{Deserialize, Serialize};

/// All training hyperparameters in one serialisable struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate for the Adam optimiser.
    pub learning_rate: f64,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Number of training epochs.
    pub num_epochs: usize,
    /// Fraction of data used for training (remainder is validation).
    pub train_ratio: f32,
    /// Where to save model checkpoints.
    pub checkpoint_dir: String,
    /// Which model size to use: "small" or "medium".
    pub model_size: String,
    /// Seed for reproducibility.
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 8,
            num_epochs: 15,
            train_ratio: 0.85,
            checkpoint_dir: "./checkpoints".to_string(),
            model_size: "small".to_string(),
            seed: 42,
        }
    }
}

impl TrainingConfig {
    /// Load from a JSON file, falling back to defaults if the file doesn't exist.
    pub fn load_or_default(path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(json) => serde_json::from_str(&json).unwrap_or_else(|e| {
                eprintln!("[WARN] Config parse error ({}): {}. Using defaults.", path, e);
                Self::default()
            }),
            Err(_) => {
                println!("[INFO] Config file '{}' not found — using defaults.", path);
                Self::default()
            }
        }
    }

    /// Save config to JSON for reproducibility.
    #[allow(dead_code)]
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}
