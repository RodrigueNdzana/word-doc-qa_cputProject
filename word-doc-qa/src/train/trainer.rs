//! Core training loop.

use std::fs;
use burn::{
    backend::{Autodiff, Wgpu},
    data::dataloader::DataLoaderBuilder,
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::Module,
    record::{CompactRecorder, Recorder},
    tensor::ElementConversion,
};

use crate::data::{dataset::QADataset, generate_qa_pairs, load_documents_from_dir, QATokenizer};
use crate::model::{ModelConfig, QAModel};
use super::{batcher::QABatcher, config::TrainingConfig, metrics::{EpochMetrics, TrainingHistory}};

pub fn run_training(docs_dir: &str, config_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = TrainingConfig::load_or_default(config_path);
    println!("[INFO] Config:\n{}", serde_json::to_string_pretty(&cfg)?);

    let documents = load_documents_from_dir(docs_dir)?;
    let pairs     = generate_qa_pairs(&documents);
    if pairs.is_empty() { return Err("No Q&A pairs generated.".into()); }

    let all_texts: Vec<String> = pairs.iter()
        .flat_map(|p| vec![p.question.clone(), p.context.clone(), p.answer.clone()])
        .collect();
    let tokenizer  = QATokenizer::build_from_texts(&all_texts);
    let vocab_size = tokenizer.vocab_size();

    fs::create_dir_all(&cfg.checkpoint_dir)?;
    fs::write(format!("{}/vocab_size.txt", cfg.checkpoint_dir), vocab_size.to_string())?;

    let full_dataset = QADataset::from_pairs(&pairs, &tokenizer);
    let (train_ds, val_ds) = full_dataset.train_val_split(cfg.train_ratio);
    println!("[INFO] Train: {}  Val: {}", train_ds.len(), val_ds.len());

    let model_cfg = if cfg.model_size == "medium" {
        ModelConfig::medium(vocab_size)
    } else {
        ModelConfig::small(vocab_size)
    };
    println!("[INFO] Model: {:?}", model_cfg);

    // Training backend = Autodiff<Wgpu>  (supports gradients)
    // Validation backend = Wgpu          (inner backend, no autodiff overhead)
    type TrainBackend = Autodiff<Wgpu>;
    type ValBackend   = Wgpu;

    let device = burn::backend::wgpu::WgpuDevice::default();

    let mut model = QAModel::<TrainBackend>::new(&device, &model_cfg);
    let mut optim = AdamConfig::new().with_epsilon(1e-8).init();

    let train_loader = DataLoaderBuilder::new(QABatcher::<TrainBackend>::new(device.clone()))
        .batch_size(cfg.batch_size).shuffle(cfg.seed).build(train_ds);
    let val_loader   = DataLoaderBuilder::new(QABatcher::<ValBackend>::new(device.clone()))
        .batch_size(cfg.batch_size).build(val_ds);

    // Two separate loss functions — one per backend type
    let train_loss_fn = CrossEntropyLossConfig::new().init::<TrainBackend>(&device);
    let val_loss_fn   = CrossEntropyLossConfig::new().init::<ValBackend>(&device);

    let mut history = TrainingHistory::default();
    let mut best_val_loss = f32::MAX;

    println!("\n══════════════════════════════════════");
    println!("  Training: {} epochs", cfg.num_epochs);
    println!("══════════════════════════════════════");

    for epoch in 0..cfg.num_epochs {
        println!("\n[Epoch {}/{}]", epoch + 1, cfg.num_epochs);

        // ── Train (Autodiff<Wgpu>) ───────────────────────────────────────
        let mut train_m = EpochMetrics::new();
        for batch in train_loader.iter() {
            let logits = model.forward(batch.input_ids.clone());
            let labels = batch.labels.clone();
            let loss   = train_loss_fn.forward(logits.clone(), labels.clone());

            let loss_f: f32 = loss.clone().into_scalar().elem();
            let pred  = logits.clone().argmax(1).squeeze::<1>();
            let ok    = pred.equal(labels.clone()).int().sum().into_scalar().elem::<i32>() as usize;
            train_m.update(loss_f, ok, batch.labels.dims()[0]);

            let grads = loss.backward();
            let gp    = GradientsParams::from_grads(grads, &model);
            model = optim.step(cfg.learning_rate, model, gp);
        }

        // ── Validate (Wgpu — inner backend, no gradients) ────────────────
        let mut val_m   = EpochMetrics::new();
        let model_valid = model.valid(); // QAModel<ValBackend>

        for batch in val_loader.iter() {
            let logits = model_valid.forward(batch.input_ids.clone());
            let labels = batch.labels.clone();
            let loss   = val_loss_fn.forward(logits.clone(), labels.clone());

            let loss_f: f32 = loss.into_scalar().elem();
            let pred  = logits.argmax(1).squeeze::<1>();
            let ok    = pred.equal(labels).int().sum().into_scalar().elem::<i32>() as usize;
            val_m.update(loss_f, ok, batch.labels.dims()[0]);
        }

        train_m.print("Train");
        val_m.print("  Val");
        history.record(&train_m, &val_m);

        // ── Checkpoint best ──────────────────────────────────────────────
        let vl = val_m.avg_loss();
        if vl < best_val_loss {
            best_val_loss = vl;
            let path = format!("{}/best_model", cfg.checkpoint_dir);
            CompactRecorder::new()
                .record(model.clone().into_record(), path.into())
                .expect("checkpoint save failed");
            println!("  ✓ Saved (val_loss={:.4})", vl);
        }
    }

    println!();
    history.print_summary();
    println!("[INFO] Done. Best val_loss={:.4}", best_val_loss);
    Ok(())
}
