//! Batcher: collates QASamples into GPU-ready tensors.
//! Burn 0.20.1 Batcher trait signature: fn batch(&self, items: Vec<I>, device: &B::Device) -> O

use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Tensor},
};
use crate::data::dataset::QASample;
use crate::data::tokenizer::MAX_SEQ_LEN;

#[derive(Clone, Debug)]
pub struct QABatch<B: Backend> {
    pub input_ids: Tensor<B, 2, burn::tensor::Int>,
    pub labels:    Tensor<B, 1, burn::tensor::Int>,
}

#[derive(Clone, Debug)]
pub struct QABatcher<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> QABatcher<B> {
    pub fn new(_device: B::Device) -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

// Burn 0.20.1: Batcher<B, I, O> with fn batch(&self, items: Vec<I>, device: &B::Device) -> O
impl<B: Backend> Batcher<B, QASample, QABatch<B>> for QABatcher<B> {
    fn batch(&self, items: Vec<QASample>, device: &B::Device) -> QABatch<B> {
        let batch_size = items.len();

        let input_flat: Vec<i32> = items
            .iter()
            .flat_map(|s| s.input_ids.iter().map(|&id| id as i32))
            .collect();

        let label_flat: Vec<i32> = items.iter().map(|s| s.label as i32).collect();

        let input_ids = Tensor::<B, 1, burn::tensor::Int>::from_ints(
            input_flat.as_slice(), device,
        )
        .reshape([batch_size, MAX_SEQ_LEN]);

        let labels = Tensor::<B, 1, burn::tensor::Int>::from_ints(
            label_flat.as_slice(), device,
        );

        QABatch { input_ids, labels }
    }
}
