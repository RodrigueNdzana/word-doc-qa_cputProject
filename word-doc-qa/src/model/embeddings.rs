use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Int, Tensor},
};

#[derive(Module, Debug)]
pub struct TokenEmbedding<B: Backend> {
    embedding: Embedding<B>,
    d_model: usize,
}

impl<B: Backend> TokenEmbedding<B> {
    pub fn new(device: &B::Device, vocab_size: usize, d_model: usize) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, d_model).init(device);
        Self { embedding, d_model }
    }

    pub fn forward(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let scale = (self.d_model as f64).sqrt() as f32;
        self.embedding.forward(token_ids) * scale
    }
}

#[derive(Module, Debug)]
pub struct PositionalEmbedding<B: Backend> {
    embedding: Embedding<B>,
}

impl<B: Backend> PositionalEmbedding<B> {
    pub fn new(device: &B::Device, max_seq_len: usize, d_model: usize) -> Self {
        let embedding = EmbeddingConfig::new(max_seq_len, d_model).init(device);
        Self { embedding }
    }

    pub fn forward(&self, batch_size: usize, seq_len: usize, device: &B::Device) -> Tensor<B, 3> {
        let positions: Vec<i32> = (0..seq_len as i32).collect();
        let pos_tensor = Tensor::<B, 1, Int>::from_ints(positions.as_slice(), device)
            .reshape([1, seq_len])
            .repeat_dim(0, batch_size);
        self.embedding.forward(pos_tensor)
    }
}
