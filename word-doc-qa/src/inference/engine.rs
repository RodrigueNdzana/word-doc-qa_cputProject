use std::fs;
use burn::{
    backend::Wgpu,
    prelude::Module,
    record::{CompactRecorder, Recorder},
    tensor::{Int, Tensor},
};
use crate::data::{load_documents_from_dir, QATokenizer};
use crate::model::{ModelConfig, QAModel};

pub fn run_inference(doc_path: &str, question: &str) -> Result<String, Box<dyn std::error::Error>> {
    let checkpoint_dir = "./checkpoints";

    // Accept both a directory path and a file path
    let actual_dir = if std::path::Path::new(doc_path).is_dir() {
        doc_path.to_string()
    } else {
        std::path::Path::new(doc_path)
            .parent()
            .and_then(|p| p.to_str())
            .unwrap_or("./data/documents")
            .to_string()
    };

    // 1. Load documents
    let documents = load_documents_from_dir(&actual_dir)?;
    let context: String = documents.iter().map(|d| d.text.as_str()).collect::<Vec<_>>().join(" ");

    // 2. If no checkpoint exists, fall back to pure keyword search
    let chk_bin  = format!("{}/best_model.bin",  checkpoint_dir);
    let chk_mpk  = format!("{}/best_model.mpk",  checkpoint_dir);
    let chk_plain = format!("{}/best_model",      checkpoint_dir);
    let checkpoint_exists =
        std::path::Path::new(&chk_bin).exists()
        || std::path::Path::new(&chk_mpk).exists()
        || std::path::Path::new(&chk_plain).exists();

    if !checkpoint_exists {
        println!("[INFO] No trained checkpoint found — using keyword search.");
        println!("[INFO] Run 'cargo run --release -- train ...' first for neural answers.");
        return Ok(retrieve_answer("", question, &context));
    }

    // 3. Rebuild tokenizer from current context
    let all_texts  = vec![question.to_string(), context.clone()];
    let tokenizer  = QATokenizer::build_from_texts(&all_texts);
    let vocab_size = tokenizer.vocab_size();

    let saved_vocab: usize = fs::read_to_string(format!("{}/vocab_size.txt", checkpoint_dir))
        .ok().and_then(|s| s.trim().parse().ok()).unwrap_or(vocab_size);

    // 4. Initialise model and load checkpoint
    type MyBackend = Wgpu;
    let device     = burn::backend::wgpu::WgpuDevice::default();
    let model_cfg  = ModelConfig::small(saved_vocab.max(vocab_size));

    let model = match CompactRecorder::new().load(chk_plain.into(), &device) {
        Ok(record) => {
            println!("[INFO] Loaded checkpoint.");
            QAModel::<MyBackend>::new(&device, &model_cfg).load_record(record)
        }
        Err(e) => {
            eprintln!("[WARN] Could not load checkpoint ({}). Using keyword search.", e);
            return Ok(retrieve_answer("", question, &context));
        }
    };

    // 5. Encode: [1, seq_len]
    let input_ids = tokenizer.encode(question, &context);
    let seq_len   = input_ids.len();
    let input_tensor = Tensor::<MyBackend, 1, Int>::from_ints(
        input_ids.iter().map(|&id| id as i32).collect::<Vec<_>>().as_slice(),
        &device,
    ).reshape([1_usize, seq_len]);

    // 6. Forward pass → logits [1, vocab_size]
    let logits = model.forward(input_tensor);

    // argmax over vocab dim → [1, 1] then flatten to scalar
    // logits is [1, vocab_size]; argmax(1) → [1]; no squeeze needed
    let top_id: u32 = logits.argmax(1).into_scalar() as u32;
    let predicted_token = tokenizer.decode(&[top_id]);

    // 7. Extractive retrieval guided by predicted token
    Ok(retrieve_answer(&predicted_token, question, &context))
}

/// Return the sentence from context most relevant to the question.
fn retrieve_answer(predicted_token: &str, question: &str, context: &str) -> String {
    let q_lower = question.to_lowercase();
    let sentences: Vec<&str> = context
        .split(|c| c == '.' || c == '\n')
        .map(|s| s.trim())
        .filter(|s| s.len() > 10)
        .collect();

    let q_words: std::collections::HashSet<&str> = q_lower
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .collect();

    let best = sentences.iter()
        .map(|&sent| {
            let s_lower = sent.to_lowercase();
            let score: usize = q_words.iter().filter(|&&w| s_lower.contains(w)).count();
            (score, sent)
        })
        .max_by_key(|(score, _)| *score);

    match best {
        Some((score, sentence)) if score > 0 => sentence.to_string(),
        _ => {
            if predicted_token.is_empty() {
                "Could not find a relevant answer in the document.".to_string()
            } else {
                format!("Predicted token: '{}'. Context: {}", predicted_token, &context[..context.len().min(300)])
            }
        }
    }
}
