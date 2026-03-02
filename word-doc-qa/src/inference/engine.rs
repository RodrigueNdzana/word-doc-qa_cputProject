use std::fs;
use burn::{
    backend::Wgpu,
    prelude::Module,
    record::{CompactRecorder, Recorder},
    tensor::{Int, Tensor},
};
use crate::data::{load_documents_from_dir, QATokenizer, generate_qa_pairs};
use crate::data::qa_pairs::QAPair;
use crate::model::{ModelConfig, QAModel};

pub fn run_inference(doc_path: &str, question: &str) -> Result<String, Box<dyn std::error::Error>> {
    let checkpoint_dir = "./checkpoints";

    let actual_dir = if std::path::Path::new(doc_path).is_dir() {
        doc_path.to_string()
    } else {
        std::path::Path::new(doc_path)
            .parent()
            .and_then(|p| p.to_str())
            .unwrap_or("./data/documents")
            .to_string()
    };

    let documents = load_documents_from_dir(&actual_dir)?;

    // ── Step 1: Search Q&A knowledge base first (always) ──────────────────
    let qa_pairs = generate_qa_pairs(&documents);
    if let Some(answer) = search_qa_pairs(&qa_pairs, question) {
        return Ok(answer);
    }

    // ── Step 2: Neural model (if checkpoint exists) ────────────────────────
    let context: String = documents.iter().map(|d| d.text.as_str()).collect::<Vec<_>>().join(" ");
    let chk_plain = format!("{}/best_model", checkpoint_dir);
    let checkpoint_exists =
        std::path::Path::new(&format!("{}.bin",  chk_plain)).exists()
            || std::path::Path::new(&format!("{}.mpk", chk_plain)).exists()
            || std::path::Path::new(&chk_plain).exists();

    if checkpoint_exists {
        let all_texts  = vec![question.to_string(), context.clone()];
        let tokenizer  = QATokenizer::build_from_texts(&all_texts);
        let vocab_size = tokenizer.vocab_size();
        let saved_vocab: usize = fs::read_to_string(format!("{}/vocab_size.txt", checkpoint_dir))
            .ok().and_then(|s| s.trim().parse().ok()).unwrap_or(vocab_size);

        type MyBackend = Wgpu;
        let device    = burn::backend::wgpu::WgpuDevice::default();
        let model_cfg = ModelConfig::small(saved_vocab.max(vocab_size));

        if let Ok(record) = CompactRecorder::new().load(chk_plain.into(), &device) {
            let model = QAModel::<MyBackend>::new(&device, &model_cfg).load_record(record);
            let input_ids = tokenizer.encode(question, &context);
            let seq_len   = input_ids.len();
            let tensor = Tensor::<MyBackend, 1, Int>::from_ints(
                input_ids.iter().map(|&id| id as i32).collect::<Vec<_>>().as_slice(),
                &device,
            ).reshape([1_usize, seq_len]);
            let logits = model.forward(tensor);
            let top_id: u32 = logits.argmax(1).into_scalar() as u32;
            let predicted = tokenizer.decode(&[top_id]);

            // Try Q&A search again using the predicted token as a hint
            if let Some(answer) = search_qa_pairs_with_hint(&qa_pairs, question, &predicted) {
                return Ok(answer);
            }
        }
    }

    // ── Step 3: Last resort — find best sentence in raw document ──────────
    Ok(find_best_sentence(question, &context))
}

// ─────────────────────────────────────────────────────────────────────────────
// Search Q&A knowledge base and return ONLY the answer string
// ─────────────────────────────────────────────────────────────────────────────
fn search_qa_pairs(pairs: &[QAPair], question: &str) -> Option<String> {
    search_qa_pairs_with_hint(pairs, question, "")
}

fn search_qa_pairs_with_hint(pairs: &[QAPair], question: &str, hint: &str) -> Option<String> {
    let q_lower = question.to_lowercase();

    let target_year = ["2024", "2025", "2026"]
        .iter()
        .find(|&&y| q_lower.contains(y))
        .copied();

    let stopwords = ["what","when","where","who","how","many","does","will",
        "the","is","are","was","were","did","do","in","on","at",
        "of","for","and","or","a","an","be","held","year","date",
        "month","their","there","that","this","which","have","has","had"];

    let q_keywords: Vec<&str> = q_lower
        .split_whitespace()
        .filter(|w| w.len() > 3 && !stopwords.contains(w))
        .collect();

    if q_keywords.is_empty() { return None; }

    let mut best_score = 0usize;
    let mut best_answer = String::new();

    for pair in pairs {
        let pq_lower = pair.question.to_lowercase();

        // Skip pairs from the wrong year
        if let Some(ty) = target_year {
            let wrong_year = ["2024","2025","2026"]
                .iter()
                .any(|&oy| oy != ty && pq_lower.contains(oy));
            if wrong_year { continue; }
        }

        let score: usize = q_keywords.iter()
            .filter(|&&kw| pq_lower.contains(kw))
            .map(|kw| kw.len())
            .sum();

        let hint_bonus = if !hint.is_empty() && pair.answer.to_lowercase().contains(&hint.to_lowercase()) { 4 } else { 0 };
        let total = score + hint_bonus;

        if total > best_score {
            best_score = total;
            best_answer = pair.answer.clone();
        }
    }

    // Require at least 2 meaningful keywords matched
    if best_score >= 6 {
        Some(best_answer)
    } else {
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Last resort: return the single best sentence from raw document text
// ─────────────────────────────────────────────────────────────────────────────
fn find_best_sentence(question: &str, context: &str) -> String {
    let q_lower = question.to_lowercase();

    let target_year = ["2024", "2025", "2026"]
        .iter()
        .find(|&&y| q_lower.contains(y))
        .copied();

    let stopwords = ["what","when","where","who","how","many","does","will",
        "the","is","are","was","were","did","do","in","on","at",
        "of","for","and","or","a","an","be","held","year","date",
        "month","their","there"];

    let q_keywords: Vec<&str> = q_lower.split_whitespace()
        .filter(|w| w.len() > 3 && !stopwords.contains(w))
        .collect();

    let sentences: Vec<&str> = context
        .split(|c| c == '.' || c == '\n')
        .map(|s| s.trim())
        .filter(|s| s.len() > 15 && s.len() < 300)
        .collect();

    let mut scored: Vec<(usize, &str)> = sentences.iter().map(|&sent| {
        let s_lower = sent.to_lowercase();
        if let Some(ty) = target_year {
            let wrong_year = ["2024","2025","2026"]
                .iter()
                .any(|&oy| oy != ty && s_lower.contains(oy));
            if wrong_year { return (0, sent); }
        }
        let score: usize = q_keywords.iter()
            .filter(|&&kw| s_lower.contains(kw))
            .map(|kw| kw.len())
            .sum();
        (score, sent)
    }).collect();

    scored.sort_by(|a, b| b.0.cmp(&a.0));

    match scored.first() {
        Some((s, sentence)) if *s > 0 => sentence.to_string(),
        _ => "No answer found in the documents.".to_string(),
    }
}