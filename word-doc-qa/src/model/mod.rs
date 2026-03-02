/// Model module: token embeddings, positional embeddings,
/// transformer encoder layers, and the full QAModel.

pub mod embeddings;
pub mod encoder;
pub mod qa_model;
pub mod config;

pub use qa_model::QAModel;
pub use config::ModelConfig;
