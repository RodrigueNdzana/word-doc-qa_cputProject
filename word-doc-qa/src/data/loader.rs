//! Document loader: reads .docx files using docx-rs and extracts plain text
//! via the JSON representation to avoid brittle enum matching.

use docx_rs::read_docx;
use serde_json::Value;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Document {
    pub filename: String,
    pub text: String,
}

/// Load all .docx files from a directory and return their extracted text.
pub fn load_documents_from_dir(dir: &str) -> Result<Vec<Document>, Box<dyn std::error::Error>> {
    let path = Path::new(dir);
    let mut documents = Vec::new();

    if path.exists() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let file_path = entry.path();
            if file_path.extension().and_then(|s| s.to_str()) == Some("docx") {
                let filename = file_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                println!("[INFO] Loading: {}", filename);
                match extract_text_from_docx(&file_path) {
                    Ok(text) => {
                        println!("[INFO]   {} characters extracted.", text.len());
                        documents.push(Document { filename, text });
                    }
                    Err(e) => eprintln!("[WARN] Skipping {}: {}", filename, e),
                }
            }
        }
    }

    if documents.is_empty() {
        println!("[INFO] No .docx files found — using built-in CPUT synthetic data.");
        documents.push(Document {
            filename: "cput_synthetic.docx".to_string(),
            text: synthetic_cput_text(),
        });
    }

    Ok(documents)
}

/// Extract all text from a .docx file by converting it to JSON and
/// recursively collecting every node where "type" == "text".
/// This is the most robust approach for docx-rs 0.4.x.
fn extract_text_from_docx(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let docx  = read_docx(&bytes)?;

    // Convert the entire document tree to JSON once
    let json_str = docx.json();
    let value: Value = serde_json::from_str(&json_str)?;

    let mut parts: Vec<String> = Vec::new();
    collect_text(&value, &mut parts);

    Ok(parts.join(" "))
}

/// Recursively walk the serde_json Value tree and collect text leaf nodes.
fn collect_text(node: &Value, out: &mut Vec<String>) {
    match node {
        Value::Object(map) => {
            // A text node looks like: {"type":"text","data":{"text":"hello"}}
            if map.get("type").and_then(|v| v.as_str()) == Some("text") {
                if let Some(text) = map
                    .get("data")
                    .and_then(|d| d.get("text"))
                    .and_then(|t| t.as_str())
                {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        out.push(trimmed.to_string());
                    }
                }
            }
            // Recurse into all values regardless
            for v in map.values() {
                collect_text(v, out);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                collect_text(v, out);
            }
        }
        _ => {}
    }
}

fn synthetic_cput_text() -> String {
    "Cape Peninsula University of Technology Academic Calendar 2024-2026. \
    The 2026 End of Year Graduation Ceremony will be held on December 12 2026. \
    The ceremony will take place at the Cape Town International Convention Centre. \
    The Higher Degrees Committee HDC held their meetings 6 times in 2024. \
    The HDC meetings were held on 15 February 2024 20 March 2024 18 April 2024 \
    22 August 2024 19 September 2024 and 14 November 2024. \
    The Cape Peninsula University of Technology CPUT was established in 2005. \
    CPUT has six faculties Applied Sciences Business and Management Sciences \
    Education and Social Sciences Engineering and the Built Environment \
    Health and Wellness Sciences and Informatics and Design. \
    The Vice-Chancellor of CPUT is Professor Chris Nhlapo. \
    CPUT has campuses in Cape Town Bellville Granger Bay and Wellington. \
    Registration for the 2025 academic year opens on January 6 2025. \
    The first semester begins on February 3 2025 and ends on June 27 2025. \
    The second semester begins on July 21 2025 and ends on November 28 2025. \
    CPUT has approximately 35000 enrolled students across all campuses. \
    The NSFAS application deadline for 2025 was November 30 2024. \
    The Research Office coordinates over 150 active research projects annually. \
    CPUT achieved a research output of 380 accredited publications in 2023."
        .to_string()
}
