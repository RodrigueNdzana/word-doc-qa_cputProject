#![allow(dead_code)]
use burn::data::dataset::Dataset;
use serde::{Deserialize, Serialize};
use super::{qa_pairs::QAPair, tokenizer::QATokenizer};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QASample {
    pub input_ids: Vec<u32>,
    pub label_ids: Vec<u32>,
    pub label: u32,
}

pub struct QADataset {
    samples: Vec<QASample>,
}

impl QADataset {
    pub fn from_pairs(pairs: &[QAPair], tokenizer: &QATokenizer) -> Self {
        let samples: Vec<QASample> = pairs
            .iter()
            .map(|pair| {
                let input_ids = tokenizer.encode(&pair.question, &pair.context);
                let label_ids = tokenizer.encode_answer(&pair.answer);
                let label = *label_ids.first().unwrap_or(&1);
                QASample { input_ids, label_ids, label }
            })
            .collect();
        QADataset { samples }
    }

    pub fn train_val_split(self, train_ratio: f32) -> (QADataset, QADataset) {
        let total = self.samples.len();
        let train_end = ((total as f32) * train_ratio).round() as usize;
        let mut all = self.samples;
        let val_samples = all.split_off(train_end);
        (QADataset { samples: all }, QADataset { samples: val_samples })
    }

    pub fn len(&self) -> usize { self.samples.len() }
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool { self.samples.is_empty() }
}

impl Dataset<QASample> for QADataset {
    fn get(&self, index: usize) -> Option<QASample> {
        self.samples.get(index).cloned()
    }
    fn len(&self) -> usize { self.samples.len() }
}
