use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Int, Tensor},
};
use super::{
    config::ModelConfig,
    embeddings::{PositionalEmbedding, TokenEmbedding},
    encoder::TransformerEncoder,
};

#[derive(Module, Debug)]
pub struct QAModel<B: Backend> {
    token_embedding:      TokenEmbedding<B>,
    positional_embedding: PositionalEmbedding<B>,
    encoder:              TransformerEncoder<B>,
    output_projection:    Linear<B>,
    d_model:              usize,
}

impl<B: Backend> QAModel<B> {
    pub fn new(device: &B::Device, config: &ModelConfig) -> Self {
        config.validate().expect("Invalid ModelConfig");
        Self {
            token_embedding:      TokenEmbedding::new(device, config.vocab_size, config.d_model),
            positional_embedding: PositionalEmbedding::new(device, config.max_seq_len, config.d_model),
            encoder:              TransformerEncoder::new(
                device, config.n_layers, config.d_model,
                config.n_heads, config.d_ff, config.dropout,
            ),
            output_projection:    LinearConfig::new(config.d_model, config.vocab_size).init(device),
            d_model:              config.d_model,
        }
    }

    /// Forward: input_ids [batch, seq_len] → logits [batch, vocab_size]
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();

        // Embeddings: [batch, seq_len, d_model]
        let tok_emb = self.token_embedding.forward(input_ids);
        let pos_emb = self.positional_embedding.forward(batch_size, seq_len, &device);
        let x       = tok_emb + pos_emb;

        // Encoder: [batch, seq_len, d_model]
        let encoded = self.encoder.forward(x);

        // Pool CLS token at position 0.
        // encoded: [batch, seq_len, d_model]
        // We want: [batch, d_model]
        //
        // Use narrow() to slice position 0 along dim 1 → [batch, 1, d_model]
        // then reshape to [batch, d_model] — avoids squeeze entirely.
        let cls = encoded
            .narrow(1, 0, 1)              // [batch, 1, d_model]
            .reshape([batch_size, self.d_model]); // [batch, d_model]

        // Project to vocab: [batch, vocab_size]
        self.output_projection.forward(cls)
    }
}
