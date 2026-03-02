/// word-doc-qa: A transformer-based Q&A system using Rust and Burn framework.
/// Supports two CLI commands:
///   - train: fine-tune the model on .docx files
///   - ask:   answer a question given a .docx document

mod data;
mod model;
mod train;
mod inference;

use std::env;

fn print_usage() {
    println!("=== word-doc-qa: Transformer Q&A System ===");
    println!();
    println!("USAGE:");
    println!("  word-doc-qa train --docs <docs_dir> --config <config.json>");
    println!("  word-doc-qa ask   --doc <file.docx>  --question \"<question>\"");
    println!();
    println!("EXAMPLES:");
    println!("  word-doc-qa train --docs ./data/documents --config config.json");
    println!("  word-doc-qa ask --doc ./data/documents/cput.docx --question \"What is the graduation date?\"");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    match args[1].as_str() {
        "train" => {
            let docs_dir = parse_arg(&args, "--docs").unwrap_or_else(|| "./data/documents".to_string());
            let config_path = parse_arg(&args, "--config").unwrap_or_else(|| "config.json".to_string());

            println!("[INFO] Starting training...");
            println!("[INFO] Docs directory : {}", docs_dir);
            println!("[INFO] Config file    : {}", config_path);

            if let Err(e) = train::run_training(&docs_dir, &config_path) {
                eprintln!("[ERROR] Training failed: {}", e);
                std::process::exit(1);
            }
        }

        "ask" => {
            let doc_path = parse_arg(&args, "--doc").unwrap_or_else(|| {
                eprintln!("[ERROR] --doc argument is required for 'ask' command.");
                std::process::exit(1);
            });
            let question = parse_arg(&args, "--question").unwrap_or_else(|| {
                eprintln!("[ERROR] --question argument is required for 'ask' command.");
                std::process::exit(1);
            });

            println!("[INFO] Loading model and answering question...");
            println!("[INFO] Document : {}", doc_path);
            println!("[INFO] Question : {}", question);
            println!();

            match inference::run_inference(&doc_path, &question) {
                Ok(answer) => {
                    println!("Answer: {}", answer);
                }
                Err(e) => {
                    eprintln!("[ERROR] Inference failed: {}", e);
                    std::process::exit(1);
                }
            }
        }

        _ => {
            eprintln!("[ERROR] Unknown command: '{}'", args[1]);
            print_usage();
            std::process::exit(1);
        }
    }
}

/// Parse a named CLI argument: --key value
fn parse_arg(args: &[String], key: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == key)
        .map(|w| w[1].clone())
}
