/// Training metrics: per-epoch loss and accuracy tracking.

/// Accumulates metrics for a single epoch.
#[derive(Default, Debug)]
pub struct EpochMetrics {
    total_loss: f32,
    correct: usize,
    total: usize,
    num_batches: usize,
}

impl EpochMetrics {
    pub fn new() -> Self { Self::default() }

    /// Record results from one batch.
    pub fn update(&mut self, loss: f32, n_correct: usize, n_total: usize) {
        self.total_loss += loss;
        self.correct += n_correct;
        self.total += n_total;
        self.num_batches += 1;
    }

    /// Average loss across batches.
    pub fn avg_loss(&self) -> f32 {
        if self.num_batches == 0 { 0.0 } else { self.total_loss / self.num_batches as f32 }
    }

    /// Top-1 accuracy.
    pub fn accuracy(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.correct as f32 / self.total as f32 }
    }

    pub fn print(&self, prefix: &str) {
        println!(
            "  {} → loss: {:.4}  accuracy: {:.2}%",
            prefix,
            self.avg_loss(),
            self.accuracy() * 100.0,
        );
    }
}

/// Full training history for all epochs (used in report generation).
#[derive(Default, Debug)]
pub struct TrainingHistory {
    pub train_losses: Vec<f32>,
    pub val_losses: Vec<f32>,
    pub train_accs: Vec<f32>,
    pub val_accs: Vec<f32>,
}

impl TrainingHistory {
    pub fn record(&mut self, train: &EpochMetrics, val: &EpochMetrics) {
        self.train_losses.push(train.avg_loss());
        self.val_losses.push(val.avg_loss());
        self.train_accs.push(train.accuracy());
        self.val_accs.push(val.accuracy());
    }

    /// Print an ASCII loss/accuracy table for the report.
    pub fn print_summary(&self) {
        println!();
        println!("┌───────┬──────────────┬──────────────┬──────────────┬──────────────┐");
        println!("│ Epoch │  Train Loss  │   Val Loss   │  Train Acc%  │   Val Acc%   │");
        println!("├───────┼──────────────┼──────────────┼──────────────┼──────────────┤");
        for (i, ((tl, vl), (ta, va))) in self.train_losses.iter()
            .zip(self.val_losses.iter())
            .zip(self.train_accs.iter().zip(self.val_accs.iter()))
            .enumerate()
        {
            println!("│  {:>4} │   {:>8.4}   │   {:>8.4}   │    {:>6.2}%   │    {:>6.2}%   │",
                i + 1, tl, vl, ta * 100.0, va * 100.0);
        }
        println!("└───────┴──────────────┴──────────────┴──────────────┴──────────────┘");
        println!();
    }
}
