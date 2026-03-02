# word-doc-qa — Transformer Q&A System in Rust + Burn

A complete machine-learning pipeline that reads `.docx` documents and answers
natural language questions about them, built with **Rust** and the **Burn**
deep-learning framework.

---

## Prerequisites

| Tool | Version |
|------|---------|
| Rust | ≥ 1.75 (stable) |
| GPU / WGPU-compatible driver | Vulkan, Metal, or DX12 |

```powershell
# Install Rust (run in PowerShell)
winget install Rustlang.Rust.MSVC
# Then restart your terminal and verify:
rustc --version
```

---

## Build

```powershell
cd word-doc-qa
cargo build --release
```

> ⚠️ **Do not modify dependency versions in `Cargo.toml`** 

---

## Quick Start

### Step 1 — Add your `.docx` files

Place your documents inside the `data/documents/` folder:

```
word-doc-qa/
└── data/
    └── documents/
        └── cput_calender.docx    ← your file goes here
```

> 💡 If no `.docx` files are present, the system automatically falls back to
> built-in synthetic CPUT data so the pipeline still runs end-to-end.

---

### Step 2 — Train the model

**Using your own `.docx` documents:**
```powershell
.\target\release\word-doc-qa.exe train --docs ./data/documents --config config.json
```

**Using the built-in synthetic CPUT data (no documents needed):**
```powershell
.\target\release\word-doc-qa.exe train --docs ./data/empty --config config.json
```

> Training takes 5–15 minutes depending on your GPU.
> The best checkpoint is saved to `./checkpoints/best_model`.
> You will see a loss/accuracy table printed when training finishes.

---

### Step 3 — Ask a question

**Answering from your own document:**
```powershell
.\target\release\word-doc-qa.exe ask --doc ./data/documents --question "What is the Month and date will the 2026 End of year Graduation Ceremony be held?"
```

```powershell
.\target\release\word-doc-qa.exe ask --doc ./data/documents --question "How many times did the HDC hold their meetings in 2024?"
```

```powershell
.\target\release\word-doc-qa.exe ask --doc ./data/documents --question "Who is the Vice-Chancellor of CPUT?"
```

> ⚠️ **PowerShell tip:** Always write the full command on a **single line**.
> Do NOT use `\` for line continuation — PowerShell does not support it.

---

## How the system decides where answers come from

| Situation | What happens |
|-----------|-------------|
| `.docx` file in `data/documents/` | Text is extracted from your document and used for training and answering |
| No `.docx` files found | Built-in synthetic CPUT data is used automatically |
| Checkpoint exists | Neural model guides answer extraction |
| No checkpoint yet | Pure keyword search is used as fallback |

---

## Configuration (`config.json`)

```json
{
  "learning_rate": 0.0001,
  "batch_size": 8,
  "num_epochs": 15,
  "train_ratio": 0.85,
  "checkpoint_dir": "./checkpoints",
  "model_size": "small",
  "seed": 42
}
```

| Key | Description | Default |
|-----|-------------|---------|
| `learning_rate` | Adam learning rate | `1e-4` |
| `batch_size` | Mini-batch size | `8` |
| `num_epochs` | Number of training epochs | `15` |
| `model_size` | `"small"` (faster) or `"medium"` (better) | `"small"` |
| `checkpoint_dir` | Where to save the trained model | `./checkpoints` |

---

## Full Workflow Example

```powershell
# 1. Build the project
cargo build --release

# 2. Copy your document into the documents folder
#    (e.g. cput_calender.docx)

# 3. Train on your document
.\target\release\word-doc-qa.exe train --docs ./data/documents --config config.json

# 4. Ask questions — answers come from your document
.\target\release\word-doc-qa.exe ask --doc ./data/documents --question "What is the Month and date will the 2026 End of year Graduation Ceremony be held?"

.\target\release\word-doc-qa.exe ask --doc ./data/documents --question "How many times did the HDC hold their meetings in 2024?"
```

---

## Project Structure

```
word-doc-qa/
├── Cargo.toml
├── config.json
├── README.md
├── docs/
│   └── report.md              ← Project report (Markdown)
├── data/
│   └── documents/             ← Place your .docx files here
├── checkpoints/               ← Created automatically during training
└── src/
    ├── main.rs                ← CLI entry point
    ├── data/
    │   ├── mod.rs
    │   ├── loader.rs          ← DOCX loading (docx-rs)
    │   ├── tokenizer.rs       ← Word-level tokenizer
    │   ├── dataset.rs         ← Burn Dataset trait impl
    │   └── qa_pairs.rs        ← Q&A pair generation
    ├── model/
    │   ├── mod.rs
    │   ├── config.rs          ← ModelConfig
    │   ├── embeddings.rs      ← Token + Positional embeddings
    │   ├── encoder.rs         ← Transformer encoder layers
    │   └── qa_model.rs        ← Full QAModel (generic over Backend)
    ├── train/
    │   ├── mod.rs
    │   ├── config.rs          ← TrainingConfig
    │   ├── batcher.rs         ← Burn Batcher impl
    │   ├── metrics.rs         ← Loss/accuracy tracking
    │   └── trainer.rs         ← Training loop + checkpointing
    └── inference/
        ├── mod.rs
        └── engine.rs          ← Model loading + Q&A inference
```

---

## Architecture Summary

```
Input IDs  [B, 256]
     │
TokenEmbedding  +  PositionalEmbedding  → [B, 256, d_model]
     │
┌────▼─────────────────────────────┐
│  TransformerEncoder (6 layers)   │
│  ┌──────────────────────────┐    │
│  │ Pre-LN + MHA + Residual  │×6  │
│  │ Pre-LN + FFN + Residual  │    │
│  └──────────────────────────┘    │
└──────────────────────────────────┘
     │
CLS pool  [B, d_model]
     │
Linear Projection  →  Logits [B, vocab_size]
```

---

## License

MIT
