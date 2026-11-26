use super::{VulkanModel, VulkanMtpState, VulkanState};
use crate::speculative::{Predictor, Target};

fn device<T>(r: kinetic_vulkan::Result<T>) -> T {
    r.unwrap_or_else(|e| panic!("Vulkan: {e}"))
}

/// Scores are the greedy tokens themselves: logits stay on the device.
impl Target for VulkanModel<'_> {
    type State = VulkanState;
    type Scores = Vec<u32>;

    fn n_embd(&self) -> usize {
        self.n_embd
    }

    fn n_vocab(&self) -> usize {
        VulkanModel::n_vocab(self)
    }

    fn set_prompt(&self, state: &mut VulkanState, prompt: bool) {
        VulkanModel::set_prompt(self, state, prompt);
    }

    fn score(
        &self,
        state: &mut VulkanState,
        tokens: &[u32],
        all_rows: bool,
    ) -> (Vec<u32>, Vec<f32>) {
        device(self.run_greedy(state, tokens, all_rows))
    }

    fn advance(&self, state: &mut VulkanState, tokens: &[u32]) -> Vec<f32> {
        device(VulkanModel::advance(self, state, tokens))
    }

    fn advance_state(&self, state: &mut VulkanState, tokens: &[u32]) {
        device(VulkanModel::advance_state(self, state, tokens));
    }

    fn logits(
        &self,
        state: &mut VulkanState,
        tokens: &[u32],
        all_rows: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let f = device(self.run(state, tokens, all_rows));
        (f.logits, f.hidden)
    }

    fn top_logits(
        &self,
        state: &mut VulkanState,
        tokens: &[u32],
        all_rows: bool,
        top: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let f = match top {
            1 => device(self.run_top1(state, tokens, all_rows)),
            _ => device(self.run(state, tokens, all_rows)),
        };
        (f.logits, f.hidden)
    }

    fn prompt_logits(
        &self,
        state: &mut VulkanState,
        tokens: &[u32],
        top: Option<usize>,
    ) -> Vec<f32> {
        if top == Some(1) {
            return device(self.run_top1(state, tokens, false)).logits;
        }
        device(self.last_logits(state, tokens))
    }

    fn greedy(&self, scores: &Vec<u32>, row: usize) -> u32 {
        scores[row]
    }

    fn checkpoint(&self, state: &mut VulkanState) {
        device(VulkanModel::checkpoint(self, state));
    }

    fn rollback(&self, state: &mut VulkanState, keep: usize) {
        device(VulkanModel::rollback(self, state, keep));
    }
}

impl Predictor for VulkanModel<'_> {
    type MtpState = VulkanMtpState;

    fn new_mtp_state(&self) -> Option<VulkanMtpState> {
        device(VulkanModel::new_mtp_state(self))
    }

    fn mtp_len(&self, state: &VulkanMtpState) -> usize {
        state.len()
    }

    fn mtp_forward(&self, state: &mut VulkanMtpState, hidden: &[f32], tokens: &[u32]) -> Vec<f32> {
        device(VulkanModel::mtp_forward(self, state, hidden, tokens))
    }

    fn draft_token(&self, state: &mut VulkanMtpState, hidden: &[f32], vocab: usize) -> u32 {
        device(VulkanModel::draft_token(self, state, hidden, vocab))
    }

    fn mtp_checkpoint(&self, state: &mut VulkanMtpState) {
        VulkanModel::mtp_checkpoint(self, state);
    }

    fn mtp_rollback(&self, state: &mut VulkanMtpState, fed: usize, keep: usize) {
        VulkanModel::mtp_rollback(self, state, fed, keep);
    }
}
