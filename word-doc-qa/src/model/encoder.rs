use burn::{
    module::Module,
    nn::{
        attention::{MultiHeadAttention, MultiHeadAttentionConfig, MhaInput},
        Dropout, DropoutConfig,
        LayerNorm, LayerNormConfig,
        Linear, LinearConfig,
    },
    tensor::{backend::Backend, Tensor, activation::gelu},
};

#[derive(Module, Debug)]
pub struct TransformerEncoderLayer<B: Backend> {
    norm1:      LayerNorm<B>,
    attention:  MultiHeadAttention<B>,
    dropout1:   Dropout,
    norm2:      LayerNorm<B>,
    ff_linear1: Linear<B>,
    ff_linear2: Linear<B>,
    dropout2:   Dropout,
}

impl<B: Backend> TransformerEncoderLayer<B> {
    pub fn new(device: &B::Device, d_model: usize, n_heads: usize, d_ff: usize, dropout: f64) -> Self {
        Self {
            norm1:      LayerNormConfig::new(d_model).init(device),
            attention:  MultiHeadAttentionConfig::new(d_model, n_heads).with_dropout(dropout).init(device),
            dropout1:   DropoutConfig::new(dropout).init(),
            norm2:      LayerNormConfig::new(d_model).init(device),
            ff_linear1: LinearConfig::new(d_model, d_ff).init(device),
            ff_linear2: LinearConfig::new(d_ff, d_model).init(device),
            dropout2:   DropoutConfig::new(dropout).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Sub-layer 1: Pre-LN + Self-Attention + residual
        let residual = x.clone();
        let normed = self.norm1.forward(x);
        let mha_out = self.attention.forward(MhaInput::self_attn(normed));
        let x = residual + self.dropout1.forward(mha_out.context);

        // Sub-layer 2: Pre-LN + FFN + residual
        let residual = x.clone();
        let normed = self.norm2.forward(x);
        let ff = gelu(self.ff_linear1.forward(normed));
        let ff = self.ff_linear2.forward(ff);
        residual + self.dropout2.forward(ff)
    }
}

#[derive(Module, Debug)]
pub struct TransformerEncoder<B: Backend> {
    layers:     Vec<TransformerEncoderLayer<B>>,
    final_norm: LayerNorm<B>,
}

impl<B: Backend> TransformerEncoder<B> {
    pub fn new(device: &B::Device, n_layers: usize, d_model: usize, n_heads: usize, d_ff: usize, dropout: f64) -> Self {
        let layers = (0..n_layers)
            .map(|_| TransformerEncoderLayer::new(device, d_model, n_heads, d_ff, dropout))
            .collect();
        Self { layers, final_norm: LayerNormConfig::new(d_model).init(device) }
    }

    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in &self.layers {
            x = layer.forward(x);
        }
        self.final_norm.forward(x)
    }
}
