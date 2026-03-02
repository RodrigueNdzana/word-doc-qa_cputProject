/// Tokenizer wrapper around the `tokenizers` crate.
/// Uses a simple whitespace/BPE tokenizer with a fixed vocabulary for portability.

use std::collections::HashMap;

/// Maximum sequence length fed into the transformer.
pub const MAX_SEQ_LEN: usize = 256;

/// Special token IDs
pub const PAD_TOKEN_ID: u32 = 0;
pub const UNK_TOKEN_ID: u32 = 1;
pub const CLS_TOKEN_ID: u32 = 2;
pub const SEP_TOKEN_ID: u32 = 3;

/// A lightweight word-level tokenizer suitable for the CPUT Q&A domain.
/// In production you would load a pretrained BPE tokenizer via `tokenizers::Tokenizer::from_file`.
pub struct QATokenizer {
    word_to_id: HashMap<String, u32>,
    id_to_word: HashMap<u32, String>,
    vocab_size: usize,
}

impl QATokenizer {
    /// Build a vocabulary from a collection of text strings.
    pub fn build_from_texts(texts: &[String]) -> Self {
        let mut word_to_id: HashMap<String, u32> = HashMap::new();
        let mut id_to_word: HashMap<u32, String> = HashMap::new();

        // Reserve special tokens
        let specials = [
            ("<PAD>", PAD_TOKEN_ID),
            ("<UNK>", UNK_TOKEN_ID),
            ("<CLS>", CLS_TOKEN_ID),
            ("<SEP>", SEP_TOKEN_ID),
        ];
        for (tok, id) in &specials {
            word_to_id.insert(tok.to_string(), *id);
            id_to_word.insert(*id, tok.to_string());
        }

        let mut next_id: u32 = 4;

        // Tokenize all texts and build vocab
        for text in texts {
            for token in Self::tokenize_raw(text) {
                if !word_to_id.contains_key(&token) {
                    word_to_id.insert(token.clone(), next_id);
                    id_to_word.insert(next_id, token);
                    next_id += 1;
                }
            }
        }

        let vocab_size = next_id as usize;
        println!("[INFO] Vocabulary size: {}", vocab_size);

        QATokenizer { word_to_id, id_to_word, vocab_size }
    }

    /// Vocabulary size (number of unique tokens).
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Encode a question+context pair into a flat token ID sequence:
    /// [CLS] question_tokens [SEP] context_tokens [SEP] [PAD...PAD]
    pub fn encode(&self, question: &str, context: &str) -> Vec<u32> {
        let q_tokens = self.tokens_to_ids(&Self::tokenize_raw(question));
        let c_tokens = self.tokens_to_ids(&Self::tokenize_raw(context));

        // Budget: 1 CLS + q + 1 SEP + c + 1 SEP
        let max_q = 60_usize;
        let max_c = MAX_SEQ_LEN - max_q - 3;

        let q_trunc: Vec<u32> = q_tokens.into_iter().take(max_q).collect();
        let c_trunc: Vec<u32> = c_tokens.into_iter().take(max_c).collect();

        let mut ids = Vec::with_capacity(MAX_SEQ_LEN);
        ids.push(CLS_TOKEN_ID);
        ids.extend_from_slice(&q_trunc);
        ids.push(SEP_TOKEN_ID);
        ids.extend_from_slice(&c_trunc);
        ids.push(SEP_TOKEN_ID);

        // Pad to MAX_SEQ_LEN
        while ids.len() < MAX_SEQ_LEN {
            ids.push(PAD_TOKEN_ID);
        }

        ids
    }

    /// Encode an answer string into token IDs (used as labels during training).
    pub fn encode_answer(&self, answer: &str) -> Vec<u32> {
        let tokens = Self::tokenize_raw(answer);
        let mut ids = self.tokens_to_ids(&tokens);
        ids.truncate(32); // answers are short
        ids
    }

    /// Decode a sequence of token IDs back into a string.
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter(|&&id| id != PAD_TOKEN_ID && id != CLS_TOKEN_ID && id != SEP_TOKEN_ID)
            .map(|id| self.id_to_word.get(id).map(String::as_str).unwrap_or("<UNK>"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Simple whitespace + punctuation tokenizer (lowercased).
    fn tokenize_raw(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| c.is_whitespace() || c == ',' || c == '"' || c == '\'' || c == ';')
            .flat_map(|word| {
                // Split off leading/trailing punctuation
                let trimmed = word.trim_matches(|c: char| {
                    c == '.' || c == '!' || c == '?' || c == ':' || c == '(' || c == ')'
                });
                if trimmed.is_empty() { vec![] } else { vec![trimmed.to_string()] }
            })
            .filter(|t| !t.is_empty())
            .collect()
    }

    fn tokens_to_ids(&self, tokens: &[String]) -> Vec<u32> {
        tokens
            .iter()
            .map(|t| *self.word_to_id.get(t).unwrap_or(&UNK_TOKEN_ID))
            .collect()
    }
}
