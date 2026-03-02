/// Training module: configuration, training loop, metrics, and checkpointing.

pub mod config;
pub mod trainer;
pub mod batcher;
pub mod metrics;

pub use trainer::run_training;
