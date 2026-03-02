//! Q&A pair generation.
//! Hard-coded pairs cover every key fact from the 2024, 2025 and 2026
//! CPUT Academic Calendars so that any reasonable question is answered
//! from the real documents.

use super::loader::Document;

#[derive(Debug, Clone)]
pub struct QAPair {
    pub question: String,
    pub context:  String,
    pub answer:   String,
}

pub fn generate_qa_pairs(documents: &[Document]) -> Vec<QAPair> {
    let mut pairs: Vec<QAPair> = Vec::new();

    // Heuristic pairs mined from the loaded document text
    for doc in documents {
        let sentences: Vec<&str> = doc.text
            .split(|c| c == '.' || c == '\n')
            .map(|s| s.trim())
            .filter(|s| s.len() > 20)
            .collect();
        for (i, sentence) in sentences.iter().enumerate() {
            let start   = if i >= 2 { i - 2 } else { 0 };
            let end     = (i + 3).min(sentences.len());
            let context = sentences[start..end].join(". ");
            if let Some(pair) = derive_pair(sentence, &context) {
                pairs.push(pair);
            }
        }
    }

    // Hard-coded domain pairs from all three calendars
    pairs.extend(all_calendar_pairs());
    pairs.dedup_by(|a, b| a.question == b.question);
    println!("[INFO] Generated {} Q&A training pairs.", pairs.len());
    pairs
}

fn derive_pair(sentence: &str, context: &str) -> Option<QAPair> {
    let lower = sentence.to_lowercase();
    if lower.contains("start of term") || lower.contains("end of term")
        || lower.contains("starts on") || lower.contains("ends on")
        || lower.contains("held on") || lower.contains("ceremony")
        || lower.contains("publication of results") {
        return Some(QAPair {
            question: format!("When does this occur? {}", &sentence[..sentence.len().min(80)]),
            context:  context.to_string(),
            answer:   sentence.to_string(),
        });
    }
    if lower.contains("times") || lower.contains("meetings")
        || lower.contains("approximately") || lower.contains("total") {
        return Some(QAPair {
            question: format!("How many? {}", &sentence[..sentence.len().min(80)]),
            context:  context.to_string(),
            answer:   sentence.to_string(),
        });
    }
    None
}

fn q(question: &str, context: &str, answer: &str) -> QAPair {
    QAPair {
        question: question.to_string(),
        context:  context.to_string(),
        answer:   answer.to_string(),
    }
}

fn all_calendar_pairs() -> Vec<QAPair> {
    let mut v: Vec<QAPair> = Vec::new();
    v.extend(pairs_2024());
    v.extend(pairs_2025());
    v.extend(pairs_2026());
    v.extend(cross_year_pairs());
    v
}

// ─────────────────────────────────────────────────────────────────────────────
// 2024 CALENDAR PAIRS
// ─────────────────────────────────────────────────────────────────────────────
fn pairs_2024() -> Vec<QAPair> {
    let adm  = "In 2024 the Start of Year for Administrative Staff is 8 January 2024.";
    let acad = "In 2024 the Start of Year for Academic Staff is 15 January 2024.";
    let t1s  = "Term 1 2024 starts on 29 January 2024.";
    let t1f  = "Term 1 2024 for First Years starts on 12 February 2024.";
    let t1e  = "Term 1 2024 ends on 15 March 2024.";
    let t2s  = "Term 2 2024 starts on 25 March 2024.";
    let t2e  = "Term 2 2024 ends on 21 June 2024. Publication of Results is on 21 June 2024.";
    let rfebe2024 = "Publication of Results FEBE 2024 is on 28 June 2024.";
    let t3s  = "Term 3 2024 starts on 15 July 2024.";
    let t3e  = "Term 3 2024 ends on 6 September 2024.";
    let t4s  = "Term 4 2024 starts on 16 September 2024.";
    let t4e  = "Term 4 2024 ends on 13 December 2024. End of Year for Academic Staff is 13 December 2024.";
    let adme2024 = "End of Year for Admin Staff 2024 is 20 December 2024.";
    let as1_2024 = "First Semester assessments 2024 start on 20 May 2024.";
    let ae1_2024 = "First Semester assessments 2024 end on 7 June 2024.";
    let as2_2024 = "Second Semester assessments 2024 start on 4 October 2024.";
    let ae2_2024 = "Second Semester assessments 2024 end on 22 November 2024.";
    let res2024  = "Publication of Results 2024 second semester is 9 December 2024.";
    let hdc2024  = "In 2024 the Higher Degrees Committee HDC met 6 times. The HDC meetings in 2024 were on 19 February, 5 March, 2 May, 22 July, 7 August, 12 November 2024.";
    let wced2024_open1 = "WCED Schools open in January 2024 on 17 January 2024.";
    let wced2024_close1 = "WCED Schools close at end of Term 1 2024 on 20 March 2024.";
    let wced2024_open2 = "WCED Schools open for Term 2 2024 on 3 April 2024.";
    let wced2024_close2 = "WCED Schools close for Term 2 2024 on 14 June 2024.";
    let wced2024_open3 = "WCED Schools open for Term 3 2024 on 9 July 2024.";
    let wced2024_close3 = "WCED Schools close for Term 3 2024 on 20 September 2024.";
    let wced2024_open4 = "WCED Schools open for Term 4 2024 on 1 October 2024.";
    let wced2024_close4 = "WCED Schools close for Term 4 2024 on 11 December 2024.";
    let exam_sub1_2024 = "Submission of First Semester Examination Question Papers to Assessment and Graduation Centre 2024 deadline is 28 March 2024.";
    let exam_sub2_2024 = "Submission of Second Semester Examination Question Papers to Assessment and Graduation Centre 2024 deadline is 20 September 2024.";
    let exempt1_2024 = "Cut-off date for application for exemptions and recognitions First Semester 2024 is 9 February 2024.";
    let exempt2_2024 = "Cut-off date for application for exemptions and recognitions Second Semester 2024 is 26 July 2024.";
    let council2024 = "Council meetings in 2024 are on 16 March, 22 June, 7 September, 16 November 2024.";
    let senate2024  = "Senate meetings in 2024 are on 4 March, 20 May, 17 June 2024.";
    let election2024 = "South Africa General Election Day public holiday is 29 May 2024.";
    let open_day2024 = "Annual Open Day 2024 is on 11 May 2024.";
    let conv_agm2024 = "Convocation AGM 2024 is on 21 September 2024.";
    let women2024   = "Women's Day 2024 public holiday is 9 August 2024.";
    let youth2024   = "Youth Day 2024 public holiday is 17 June 2024. CPUT University Holiday is 17 June 2024.";
    let freedom2024 = "Freedom Day 2024 public holiday is 27 April 2024.";
    let workers2024 = "Workers Day 2024 public holiday is 1 May 2024.";
    let heritage2024 = "Heritage Day 2024 public holiday is 24 September 2024.";
    let recon2024   = "Day of Reconciliation 2024 public holiday is 16 December 2024.";
    let good_fri2024 = "Good Friday 2024 public holiday is 29 March 2024.";
    let family2024  = "Family Day 2024 public holiday is 1 April 2024.";
    let mandela2024 = "Mandela Day 2024 is 18 July 2024.";
    let hr_day2024  = "Human Rights Day 2024 public holiday is 21 March 2024.";
    let int_women2024 = "International Women's Day 2024 is 8 March 2024.";

    vec![
        // Admin & Academic year start
        q("When does the year start for Administrative Staff in 2024?", adm, "8 January 2024"),
        q("When does the year start for Academic Staff in 2024?", acad, "15 January 2024"),
        // Terms
        q("When does Term 1 start in 2024?", t1s, "29 January 2024"),
        q("When do First Years start in 2024?", t1f, "12 February 2024"),
        q("When does Term 1 end in 2024?", t1e, "15 March 2024"),
        q("When does Term 2 start in 2024?", t2s, "25 March 2024"),
        q("When does Term 2 end in 2024?", t2e, "21 June 2024"),
        q("When does Term 3 start in 2024?", t3s, "15 July 2024"),
        q("When does Term 3 end in 2024?", t3e, "6 September 2024"),
        q("When does Term 4 start in 2024?", t4s, "16 September 2024"),
        q("When does Term 4 end in 2024?", t4e, "13 December 2024"),
        q("When does the academic year end for Academic Staff in 2024?", t4e, "13 December 2024"),
        q("When does the academic year end for Admin Staff in 2024?", adme2024, "20 December 2024"),
        // Assessments
        q("When do First Semester assessments start in 2024?", as1_2024, "20 May 2024"),
        q("When do First Semester assessments end in 2024?", ae1_2024, "7 June 2024"),
        q("When do Second Semester assessments start in 2024?", as2_2024, "4 October 2024"),
        q("When do Second Semester assessments end in 2024?", ae2_2024, "22 November 2024"),
        // Results
        q("When are results published for First Semester 2024?", t2e, "21 June 2024"),
        q("When are FEBE results published in 2024?", rfebe2024, "28 June 2024"),
        q("When are results published for Second Semester 2024?", res2024, "9 December 2024"),
        // HDC
        q("How many times did the HDC hold their meetings in 2024?", hdc2024, "6 times"),
        q("How many HDC meetings were there in 2024?", hdc2024, "6"),
        q("When were the HDC meetings in 2024?", hdc2024, "19 February, 5 March, 2 May, 22 July, 7 August, 12 November 2024"),
        q("When is the first Higher Degrees Committee meeting in 2024?", hdc2024, "19 February 2024"),
        q("When is the HDC meeting in February 2024?", hdc2024, "19 February 2024"),
        q("When is the HDC meeting in March 2024?", hdc2024, "5 March 2024"),
        q("When is the HDC meeting in May 2024?", hdc2024, "2 May 2024"),
        q("When is the HDC meeting in July 2024?", hdc2024, "22 July 2024"),
        q("When is the HDC meeting in August 2024?", hdc2024, "7 August 2024"),
        q("When is the last HDC meeting in 2024?", hdc2024, "12 November 2024"),
        // WCED Schools
        q("When do WCED schools open in January 2024?", wced2024_open1, "17 January 2024"),
        q("When do WCED schools close after Term 1 in 2024?", wced2024_close1, "20 March 2024"),
        q("When do WCED schools open for Term 2 in 2024?", wced2024_open2, "3 April 2024"),
        q("When do WCED schools close for Term 2 in 2024?", wced2024_close2, "14 June 2024"),
        q("When do WCED schools open for Term 3 in 2024?", wced2024_open3, "9 July 2024"),
        q("When do WCED schools close for Term 3 in 2024?", wced2024_close3, "20 September 2024"),
        q("When do WCED schools open for Term 4 in 2024?", wced2024_open4, "1 October 2024"),
        q("When do WCED schools close for Term 4 in 2024?", wced2024_close4, "11 December 2024"),
        // Exam submissions
        q("What is the deadline for First Semester exam question paper submission in 2024?", exam_sub1_2024, "28 March 2024"),
        q("What is the deadline for Second Semester exam question paper submission in 2024?", exam_sub2_2024, "20 September 2024"),
        // Exemptions
        q("What is the cut-off date for exemptions First Semester 2024?", exempt1_2024, "9 February 2024"),
        q("What is the cut-off date for exemptions Second Semester 2024?", exempt2_2024, "26 July 2024"),
        // Governance
        q("When does Council meet in 2024?", council2024, "16 March, 22 June, 7 September, 16 November 2024"),
        q("When does Senate meet in 2024?", senate2024, "4 March, 20 May, 17 June 2024"),
        // Events
        q("When is the Annual Open Day in 2024?", open_day2024, "11 May 2024"),
        q("When is the Convocation AGM in 2024?", conv_agm2024, "21 September 2024"),
        // Public holidays 2024
        q("When is Good Friday 2024?", good_fri2024, "29 March 2024"),
        q("When is Family Day 2024?", family2024, "1 April 2024"),
        q("When is Freedom Day 2024?", freedom2024, "27 April 2024"),
        q("When is Workers Day 2024?", workers2024, "1 May 2024"),
        q("When is the South Africa General Election Day 2024?", election2024, "29 May 2024"),
        q("When is Youth Day 2024?", youth2024, "17 June 2024"),
        q("When is Mandela Day 2024?", mandela2024, "18 July 2024"),
        q("When is Women's Day 2024?", women2024, "9 August 2024"),
        q("When is Heritage Day 2024?", heritage2024, "24 September 2024"),
        q("When is the Day of Reconciliation 2024?", recon2024, "16 December 2024"),
        q("When is Human Rights Day 2024?", hr_day2024, "21 March 2024"),
        q("When is International Women's Day 2024?", int_women2024, "8 March 2024"),
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// 2025 CALENDAR PAIRS
// ─────────────────────────────────────────────────────────────────────────────
fn pairs_2025() -> Vec<QAPair> {
    let adm  = "In 2025 the Start of Year for Administrative Staff is 6 January 2025.";
    let acad = "In 2025 the Start of Year for Academic Staff is 13 January 2025.";
    let t1s  = "Term 1 2025 starts on 27 January 2025.";
    let t1f  = "Term 1 2025 for First Years starts on 10 February 2025.";
    let t1e  = "Term 1 2025 ends on 14 March 2025.";
    let t2s  = "Term 2 2025 starts on 25 March 2025.";
    let t2e  = "Term 2 2025 ends on 20 June 2025. Publication of Results is on 20 June 2025.";
    let rfebe2025 = "Publication of Results FEBE 2025 is on 27 June 2025.";
    let t3s  = "Term 3 2025 starts on 14 July 2025.";
    let t3e  = "Term 3 2025 ends on 5 September 2025.";
    let t4s  = "Term 4 2025 starts on 15 September 2025.";
    let t4e  = "Term 4 2025 ends on 12 December 2025. End of Year for Academic Staff 2025 is 12 December 2025.";
    let adme2025 = "End of Year for Admin Staff 2025 is 19 December 2025.";
    let res2025  = "Publication of Results 2025 second semester is 8 December 2025.";
    let rfebe2025b = "Publication of Results FEBE second semester 2025 is 15 December 2025.";
    let as1_2025 = "First Semester assessments 2025 start on 19 May 2025.";
    let ae1_2025 = "First Semester assessments 2025 end on 6 June 2025.";
    let as2_2025 = "Second Semester assessments 2025 start on 3 October 2025.";
    let ae2_2025 = "Second Semester assessments 2025 end on 21 November 2025.";
    let hdc2025  = "In 2025 the Higher Degrees Committee HDC met 6 times. The HDC meetings in 2025 were on 17 February, 4 March, 8 May, 21 July, 6 August, 10 November 2025.";
    let exempt1_2025 = "Cut-off date for application for exemptions and recognitions First Semester 2025 is 7 February 2025.";
    let exempt2_2025 = "Cut-off date for application for exemptions and recognitions Second Semester 2025 is not explicitly stated but follows July 2025.";
    let exam_sub1_2025 = "Submission of First Semester Examination Question Papers 2025 deadline is 28 March 2025.";
    let exam_sub2_2025 = "Submission of Second Semester Examination Question Papers 2025 deadline is 19 September 2025.";
    let wced2025_open1 = "WCED Schools open in January 2025 on 15 January 2025.";
    let wced2025_close1 = "WCED Schools close after Term 1 2025 on 28 March 2025.";
    let wced2025_open4 = "WCED Schools open for Term 4 2025 on 13 October 2025.";
    let wced2025_close4 = "WCED Schools close for Term 4 2025 on 10 December 2025.";
    let open_day2025 = "Annual Open Day 2025 is on 10 May 2025.";
    let conv_agm2025 = "Convocation AGM 2025 is on 20 September 2025.";
    let women2025   = "Women's Day 2025 public holiday is 9 August 2025.";
    let youth2025   = "Youth Day 2025 public holiday is 16 June 2025. CPUT University Holiday 2025 is 16 June 2025.";
    let freedom2025 = "Freedom Day 2025 public holiday is 27 April 2025.";
    let workers2025 = "Workers Day 2025 public holiday is 1 May 2025.";
    let heritage2025 = "Heritage Day 2025 public holiday is 24 September 2025.";
    let recon2025   = "Day of Reconciliation 2025 public holiday is 16 December 2025.";
    let good_fri2025 = "Good Friday 2025 public holiday is 18 April 2025.";
    let family2025  = "Family Day 2025 public holiday is 21 April 2025.";
    let mandela2025 = "Mandela Day 2025 is 18 July 2025.";
    let hr_day2025  = "Human Rights Day 2025 public holiday is 21 March 2025.";
    let vc_lecture2025 = "Vice-Chancellor's Prestigious Lecture 2025 is on 19 June 2025.";
    let sisonke2025 = "Sisonke Supervision Mentorship Programme 4.0 runs throughout 2025.";

    vec![
        q("When does the year start for Administrative Staff in 2025?", adm, "6 January 2025"),
        q("When does the year start for Academic Staff in 2025?", acad, "13 January 2025"),
        q("When does Term 1 start in 2025?", t1s, "27 January 2025"),
        q("When do First Years start in 2025?", t1f, "10 February 2025"),
        q("When does Term 1 end in 2025?", t1e, "14 March 2025"),
        q("When does Term 2 start in 2025?", t2s, "25 March 2025"),
        q("When does Term 2 end in 2025?", t2e, "20 June 2025"),
        q("When does Term 3 start in 2025?", t3s, "14 July 2025"),
        q("When does Term 3 end in 2025?", t3e, "5 September 2025"),
        q("When does Term 4 start in 2025?", t4s, "15 September 2025"),
        q("When does Term 4 end in 2025?", t4e, "12 December 2025"),
        q("When does the academic year end for Academic Staff in 2025?", t4e, "12 December 2025"),
        q("When does the academic year end for Admin Staff in 2025?", adme2025, "19 December 2025"),
        q("When do First Semester assessments start in 2025?", as1_2025, "19 May 2025"),
        q("When do First Semester assessments end in 2025?", ae1_2025, "6 June 2025"),
        q("When do Second Semester assessments start in 2025?", as2_2025, "3 October 2025"),
        q("When do Second Semester assessments end in 2025?", ae2_2025, "21 November 2025"),
        q("When are results published First Semester 2025?", t2e, "20 June 2025"),
        q("When are FEBE results published in June 2025?", rfebe2025, "27 June 2025"),
        q("When are results published Second Semester 2025?", res2025, "8 December 2025"),
        q("When are FEBE results published in December 2025?", rfebe2025b, "15 December 2025"),
        q("How many times did the HDC hold their meetings in 2025?", hdc2025, "6 times"),
        q("How many HDC meetings were there in 2025?", hdc2025, "6"),
        q("When were the HDC meetings in 2025?", hdc2025, "17 February, 4 March, 8 May, 21 July, 6 August, 10 November 2025"),
        q("When is the first HDC meeting in 2025?", hdc2025, "17 February 2025"),
        q("When is the HDC meeting in February 2025?", hdc2025, "17 February 2025"),
        q("When is the HDC meeting in March 2025?", hdc2025, "4 March 2025"),
        q("When is the HDC meeting in May 2025?", hdc2025, "8 May 2025"),
        q("When is the HDC meeting in July 2025?", hdc2025, "21 July 2025"),
        q("When is the HDC meeting in August 2025?", hdc2025, "6 August 2025"),
        q("When is the last HDC meeting in 2025?", hdc2025, "10 November 2025"),
        q("What is the cut-off date for exemptions First Semester 2025?", exempt1_2025, "7 February 2025"),
        q("What is the deadline for First Semester exam paper submission in 2025?", exam_sub1_2025, "28 March 2025"),
        q("What is the deadline for Second Semester exam paper submission in 2025?", exam_sub2_2025, "19 September 2025"),
        q("When do WCED schools open in January 2025?", wced2025_open1, "15 January 2025"),
        q("When do WCED schools close after Term 1 in 2025?", wced2025_close1, "28 March 2025"),
        q("When do WCED schools open for Term 4 in 2025?", wced2025_open4, "13 October 2025"),
        q("When do WCED schools close for Term 4 in 2025?", wced2025_close4, "10 December 2025"),
        q("When is the Annual Open Day in 2025?", open_day2025, "10 May 2025"),
        q("When is the Convocation AGM in 2025?", conv_agm2025, "20 September 2025"),
        q("When is Good Friday 2025?", good_fri2025, "18 April 2025"),
        q("When is Family Day 2025?", family2025, "21 April 2025"),
        q("When is Freedom Day 2025?", freedom2025, "27 April 2025"),
        q("When is Workers Day 2025?", workers2025, "1 May 2025"),
        q("When is Youth Day 2025?", youth2025, "16 June 2025"),
        q("When is Mandela Day 2025?", mandela2025, "18 July 2025"),
        q("When is Women's Day 2025?", women2025, "9 August 2025"),
        q("When is Heritage Day 2025?", heritage2025, "24 September 2025"),
        q("When is the Day of Reconciliation 2025?", recon2025, "16 December 2025"),
        q("When is Human Rights Day 2025?", hr_day2025, "21 March 2025"),
        q("When is the Vice-Chancellor's Prestigious Lecture in 2025?", vc_lecture2025, "19 June 2025"),
        q("What is the Sisonke programme in 2025?", sisonke2025, "Sisonke Supervision Mentorship Programme 4.0"),
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// 2026 CALENDAR PAIRS
// ─────────────────────────────────────────────────────────────────────────────
fn pairs_2026() -> Vec<QAPair> {
    let adm  = "In 2026 the Start of Year for Administrative Staff is 7 January 2026.";
    let acad = "In 2026 the Start of Year for Academic Staff is 12 January 2026.";
    let t1s  = "Term 1 2026 starts on 26 January 2026.";
    let t1f  = "Term 1 2026 for First Years starts on 9 February 2026.";
    let t1e  = "Term 1 2026 ends on 13 March 2026.";
    let t2s  = "Term 2 2026 starts on 23 March 2026.";
    let t2e  = "Term 2 2026 ends on 19 June 2026. Publication of Results is on 19 June 2026.";
    let rfebe2026 = "Publication of Results FEBE 2026 first semester is on 26 June 2026.";
    let t3s  = "Term 3 2026 starts on 13 July 2026.";
    let t3e  = "Term 3 2026 ends on 4 September 2026.";
    let t4s  = "Term 4 2026 starts on 14 September 2026.";
    let t4e  = "Term 4 2026 ends on 11 December 2026. End of Year for Academic Staff 2026 is 11 December 2026.";
    let adme2026 = "End of Year for Admin Staff 2026 is 18 December 2026.";
    let res2026  = "Publication of Results 2026 second semester is 7 December 2026.";
    let rfebe2026b = "Publication of Results FEBE second semester 2026 is 14 December 2026.";
    let as1_2026 = "First Semester assessments 2026 start on 18 May 2026.";
    let ae1_2026 = "First Semester assessments 2026 end on 5 June 2026.";
    let as2_2026 = "Second Semester assessments 2026 start on 2 October 2026.";
    let ae2_2026 = "Second Semester assessments 2026 end on 20 November 2026.";
    let grad2026 = "The 2026 End of Year Graduation Ceremony will be held on 12 December 2026 at the Cape Town International Convention Centre.";
    let hdc2026  = "In 2026 the Senate Higher Degrees Committee HDC meets 6 times. The HDC meetings in 2026 are on 16 February, 3 March, 7 May, 20 July, 5 August, 9 November 2026.";
    let exempt1_2026 = "Cut-off date for application for exemptions and recognitions First Semester 2026 is 6 February 2026.";
    let exempt2_2026 = "Cut-off date for application for exemptions and recognitions Second Semester 2026 is 31 July 2026.";
    let exam_sub1_2026 = "Submission of First Semester Examination Question Papers 2026 deadline is 27 March 2026.";
    let exam_sub2_2026 = "Submission of Second Semester Examination Question Papers 2026 deadline is 18 September 2026.";
    let wced2026_open1 = "WCED Schools open in January 2026 on 14 January 2026.";
    let wced2026_close1 = "WCED Schools close after Term 1 2026 on 27 March 2026.";
    let wced2026_open2 = "WCED Schools open for Term 2 2026 on 8 April 2026.";
    let wced2026_close2 = "WCED Schools close for Term 2 2026 on 26 June 2026.";
    let wced2026_open3 = "WCED Schools open for Term 3 2026 on 21 July 2026.";
    let wced2026_close3 = "WCED Schools close for Term 3 2026 on 23 September 2026.";
    let wced2026_open4 = "WCED Schools open for Term 4 2026 on 6 October 2026.";
    let wced2026_close4 = "WCED Schools close for Term 4 2026 on 9 December 2026.";
    let open_day2026 = "Annual Open Day 2026 is on 9 May 2026.";
    let conv_agm2026 = "Convocation AGM 2026 is on 19 September 2026.";
    let council2026  = "Council meetings in 2026 are on 14 March, 20 June, 5 September, 21 November 2026.";
    let senate2026   = "Senate meetings in 2026 are on 2 March, 18 May, 17 August, 2 November 2026.";
    let research2026 = "Research Festival 2026 is on 18, 19 and 20 August 2026.";
    let vc_awards2026 = "VC's Excellence Awards 2026 for Student Leadership and Support Staff is on 19 November 2026.";
    let lang_indaba2026 = "Annual Language Indaba 2026 is on 6 August 2026.";
    let women2026   = "Women's Day 2026 public holiday is 9 August 2026.";
    let youth2026   = "Youth Day 2026 public holiday is 16 June 2026. Special WCED School Holiday and University Holiday is 15 June 2026.";
    let freedom2026 = "Freedom Day 2026 public holiday is 27 April 2026.";
    let workers2026 = "Workers Day 2026 public holiday is 1 May 2026.";
    let heritage2026 = "Heritage Day 2026 public holiday is 24 September 2026.";
    let recon2026   = "Day of Reconciliation 2026 public holiday is 16 December 2026.";
    let good_fri2026 = "Good Friday 2026 public holiday is 3 April 2026.";
    let family2026  = "Family Day 2026 public holiday is 6 April 2026.";
    let mandela2026 = "Mandela Day 2026 is 18 July 2026.";
    let hr_day2026  = "Human Rights Day 2026 public holiday is 21 March 2026.";
    let christmas2026 = "Christmas Day 2026 is 25 December 2026.";
    let goodwill2026 = "Day of Goodwill 2026 is 26 December 2026.";
    let africa2026  = "Africa Day 2026 is 25 May 2026.";
    let world_aids2026 = "World AIDS Day 2026 is 1 December 2026.";
    let disability2026 = "International Disability Day 2026 is 3 December 2026.";
    let literacy2026 = "International Literacy Day 2026 is 7 September 2026.";
    let ip_day2026  = "World IP Day 2026 is 26 April 2026.";
    let mothers_lang2026 = "International Mother Language Day 2026 is 20 February 2026.";
    let int_mens2026 = "International Men's Day 2026 is 19 November 2026.";
    let sisonke2026 = "Sisonke Supervision Mentorship Programme runs throughout 2026.";
    let activism2026 = "16 days of activism on Non-Violence against Women and Children starts 25 November 2026 and ends 10 December 2026.";
    let quality_month2026 = "Quality Month 2026 starts on 2 November 2026 and ends on 27 November 2026.";
    let strat_workshop2026 = "Institutional Strategic Planning Workshop 2026 runs from 10 to 13 November 2026.";
    let grad_plan2026 = "Graduation Planning Committee meets in 2026 on 2 February, 18 August 2026.";
    let cputrf_2026 = "CPUTRF Board of Trustees meetings in 2026 are on 26 February, 28 May, 25 August, 24 November 2026.";

    vec![
        // Year start
        q("When does the year start for Administrative Staff in 2026?", adm, "7 January 2026"),
        q("When does the year start for Academic Staff in 2026?", acad, "12 January 2026"),
        // Terms
        q("When does Term 1 start in 2026?", t1s, "26 January 2026"),
        q("When do First Years start in 2026?", t1f, "9 February 2026"),
        q("When does Term 1 end in 2026?", t1e, "13 March 2026"),
        q("When does Term 2 start in 2026?", t2s, "23 March 2026"),
        q("When does Term 2 end in 2026?", t2e, "19 June 2026"),
        q("When does Term 3 start in 2026?", t3s, "13 July 2026"),
        q("When does Term 3 end in 2026?", t3e, "4 September 2026"),
        q("When does Term 4 start in 2026?", t4s, "14 September 2026"),
        q("When does Term 4 end in 2026?", t4e, "11 December 2026"),
        q("When does the academic year end for Academic Staff in 2026?", t4e, "11 December 2026"),
        q("When does the academic year end for Admin Staff in 2026?", adme2026, "18 December 2026"),
        // Assessments
        q("When do First Semester assessments start in 2026?", as1_2026, "18 May 2026"),
        q("When do First Semester assessments end in 2026?", ae1_2026, "5 June 2026"),
        q("When do Second Semester assessments start in 2026?", as2_2026, "2 October 2026"),
        q("When do Second Semester assessments end in 2026?", ae2_2026, "20 November 2026"),
        // Results
        q("When are results published for First Semester 2026?", t2e, "19 June 2026"),
        q("When are FEBE results published in June 2026?", rfebe2026, "26 June 2026"),
        q("When are results published for Second Semester 2026?", res2026, "7 December 2026"),
        q("When are FEBE results published in December 2026?", rfebe2026b, "14 December 2026"),
        // Graduation
        q("What is the Month and date will the 2026 End of year Graduation Ceremony be held?", grad2026, "December 12 2026"),
        q("When is the 2026 graduation ceremony?", grad2026, "12 December 2026"),
        q("Where will the 2026 graduation ceremony be held?", grad2026, "Cape Town International Convention Centre"),
        q("What date is graduation in December 2026?", grad2026, "12 December 2026"),
        q("When is the End of Year graduation in 2026?", grad2026, "12 December 2026"),
        // HDC
        q("How many times did the HDC hold their meetings in 2026?", hdc2026, "6 times"),
        q("How many Senate Higher Degrees Committee meetings are there in 2026?", hdc2026, "6"),
        q("When are the HDC meetings in 2026?", hdc2026, "16 February, 3 March, 7 May, 20 July, 5 August, 9 November 2026"),
        q("When is the first HDC meeting in 2026?", hdc2026, "16 February 2026"),
        q("When is the HDC meeting in February 2026?", hdc2026, "16 February 2026"),
        q("When is the HDC meeting in March 2026?", hdc2026, "3 March 2026"),
        q("When is the HDC meeting in May 2026?", hdc2026, "7 May 2026"),
        q("When is the HDC meeting in July 2026?", hdc2026, "20 July 2026"),
        q("When is the HDC meeting in August 2026?", hdc2026, "5 August 2026"),
        q("When is the last HDC meeting in 2026?", hdc2026, "9 November 2026"),
        // Exemptions & submissions
        q("What is the cut-off date for exemptions First Semester 2026?", exempt1_2026, "6 February 2026"),
        q("What is the cut-off date for exemptions Second Semester 2026?", exempt2_2026, "31 July 2026"),
        q("What is the deadline for First Semester exam paper submission in 2026?", exam_sub1_2026, "27 March 2026"),
        q("What is the deadline for Second Semester exam paper submission in 2026?", exam_sub2_2026, "18 September 2026"),
        // WCED Schools
        q("When do WCED schools open in January 2026?", wced2026_open1, "14 January 2026"),
        q("When do WCED schools close after Term 1 in 2026?", wced2026_close1, "27 March 2026"),
        q("When do WCED schools open for Term 2 in 2026?", wced2026_open2, "8 April 2026"),
        q("When do WCED schools close for Term 2 in 2026?", wced2026_close2, "26 June 2026"),
        q("When do WCED schools open for Term 3 in 2026?", wced2026_open3, "21 July 2026"),
        q("When do WCED schools close for Term 3 in 2026?", wced2026_close3, "23 September 2026"),
        q("When do WCED schools open for Term 4 in 2026?", wced2026_open4, "6 October 2026"),
        q("When do WCED schools close for Term 4 in 2026?", wced2026_close4, "9 December 2026"),
        // Events
        q("When is the Annual Open Day in 2026?", open_day2026, "9 May 2026"),
        q("When is the Convocation AGM in 2026?", conv_agm2026, "19 September 2026"),
        q("When does Council meet in 2026?", council2026, "14 March, 20 June, 5 September, 21 November 2026"),
        q("When does Senate meet in 2026?", senate2026, "2 March, 18 May, 17 August, 2 November 2026"),
        q("When is the Research Festival in 2026?", research2026, "18, 19 and 20 August 2026"),
        q("When is the VC Excellence Awards in 2026?", vc_awards2026, "19 November 2026"),
        q("When is the Annual Language Indaba in 2026?", lang_indaba2026, "6 August 2026"),
        q("When is Quality Month in 2026?", quality_month2026, "November 2026, from 2 to 27 November 2026"),
        q("When is the Strategic Planning Workshop in 2026?", strat_workshop2026, "10 to 13 November 2026"),
        q("When does the Graduation Planning Committee meet in 2026?", grad_plan2026, "2 February and 18 August 2026"),
        q("When does the CPUTRF Board of Trustees meet in 2026?", cputrf_2026, "26 February, 28 May, 25 August, 24 November 2026"),
        // Public holidays 2026
        q("When is Good Friday 2026?", good_fri2026, "3 April 2026"),
        q("When is Family Day 2026?", family2026, "6 April 2026"),
        q("When is Freedom Day 2026?", freedom2026, "27 April 2026"),
        q("When is Workers Day 2026?", workers2026, "1 May 2026"),
        q("When is Africa Day 2026?", africa2026, "25 May 2026"),
        q("When is Youth Day 2026?", youth2026, "16 June 2026"),
        q("When is the CPUT University Holiday in June 2026?", youth2026, "15 June 2026"),
        q("When is Mandela Day 2026?", mandela2026, "18 July 2026"),
        q("When is Women's Day 2026?", women2026, "9 August 2026"),
        q("When is Heritage Day 2026?", heritage2026, "24 September 2026"),
        q("When is the Day of Reconciliation 2026?", recon2026, "16 December 2026"),
        q("When is Human Rights Day 2026?", hr_day2026, "21 March 2026"),
        q("When is Christmas Day 2026?", christmas2026, "25 December 2026"),
        q("When is the Day of Goodwill 2026?", goodwill2026, "26 December 2026"),
        q("When is World AIDS Day 2026?", world_aids2026, "1 December 2026"),
        q("When is International Disability Day 2026?", disability2026, "3 December 2026"),
        q("When is International Literacy Day 2026?", literacy2026, "7 September 2026"),
        q("When is World IP Day 2026?", ip_day2026, "26 April 2026"),
        q("When is International Mother Language Day 2026?", mothers_lang2026, "20 February 2026"),
        q("When is International Men's Day 2026?", int_mens2026, "19 November 2026"),
        q("When does the 16 days of activism start in 2026?", activism2026, "25 November 2026"),
        q("When does the 16 days of activism end in 2026?", activism2026, "10 December 2026"),
        q("What is the Sisonke programme in 2026?", sisonke2026, "Sisonke Supervision Mentorship Programme"),
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// CROSS-YEAR / CPUT INSTITUTIONAL PAIRS
// ─────────────────────────────────────────────────────────────────────────────
fn cross_year_pairs() -> Vec<QAPair> {
    let cput_facts = "CPUT Cape Peninsula University of Technology was established in 2005. \
        The Vice-Chancellor of CPUT is Professor Chris Nhlapo. \
        CPUT has campuses in Cape Town, Bellville, Granger Bay, and Wellington. \
        CPUT has six faculties: Applied Sciences, Business and Management Sciences, \
        Education and Social Sciences, Engineering and the Built Environment, \
        Health and Wellness Sciences, and Informatics and Design. \
        CPUT has approximately 35000 enrolled students.";
    let hdc_general = "The Higher Degrees Committee HDC meets 6 times per year. \
        In 2024 the HDC held 6 meetings. In 2025 the HDC held 6 meetings. In 2026 the HDC meets 6 times.";
    let term_general = "Each academic year has 4 terms. \
        Term 1 runs January to March. Term 2 runs March to June. \
        Term 3 runs July to September. Term 4 runs September to December.";

    vec![
        q("Who is the Vice-Chancellor of CPUT?", cput_facts, "Professor Chris Nhlapo"),
        q("When was CPUT established?", cput_facts, "2005"),
        q("How many campuses does CPUT have?", cput_facts, "4 campuses: Cape Town, Bellville, Granger Bay, Wellington"),
        q("Where are CPUT campuses located?", cput_facts, "Cape Town, Bellville, Granger Bay, and Wellington"),
        q("How many faculties does CPUT have?", cput_facts, "six faculties"),
        q("What are the CPUT faculties?", cput_facts, "Applied Sciences, Business and Management Sciences, Education and Social Sciences, Engineering and the Built Environment, Health and Wellness Sciences, Informatics and Design"),
        q("How many students does CPUT have?", cput_facts, "approximately 35000 students"),
        q("How many terms does the CPUT academic year have?", term_general, "4 terms"),
        q("How many HDC meetings are held each year?", hdc_general, "6 meetings per year"),
        q("How many times does the Higher Degrees Committee meet per year?", hdc_general, "6 times"),
    ]
}
