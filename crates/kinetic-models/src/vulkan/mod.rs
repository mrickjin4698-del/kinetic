//! Whole models on a Vulkan GPU: AMD, Intel or NVIDIA.
//!
//! [`VulkanModel::load`] uploads the weights of a loaded [`Model`] once; a
//! [`VulkanState`] holds a sequence's caches and working buffers on the
//! device. Token embeddings and rotary angles are computed on the host,
//! exactly as the CPU path computes them; greedy tokens, hidden states and,
//! when asked for, logits come back.
//!
//! A state records its pass once, as a command buffer replayed for every
//! batch of up to [`MAX_BATCH`] tokens: the kernels read how many tokens run
//! and where from a small buffer the host fills before each run. Every
//! kernel computes a token's result the same way whatever else shares its
//! batch, so a verified speculative draft yields exactly the tokens plain
//! decoding would. Attention caches and convolution rings are indexed by
//! position, so a rollback has nothing to undo there; delta-rule layers
//! leave their state alone after a checkpoint and replay the kept tokens
//! from a journal on rollback.

mod attention;
mod delta_net;
mod ffn;
mod host;
mod layer;
mod mtp;
mod rope;
mod short_conv;
mod target;
mod work;

use std::ops::Range;
use std::rc::Rc;

use kinetic_quant::{dequantize_row, Matrix};
use kinetic_vulkan::{
    Batch, Commands, Device, DeviceBuffer, DeviceMatrix, Graph, Instance, Kernels, Out,
    PhysicalDevice, Rows,
};

use crate::model::{embed_into, Mark};
use crate::{Forward, LoadError, Model};
use layer::{LayerState, VulkanLayer};
use mtp::VulkanMtp;

pub use host::VulkanRunner;
pub use mtp::VulkanMtpState;
use rope::RopeTables;
use work::Work;

/// Most tokens one pass runs, prompt passes apart, and most that may run
/// between a checkpoint and its rollback.
pub const MAX_BATCH: usize = 16;

/// Fewest tokens worth a prompt pass: below, the kernels a token is
/// decoded with are the faster ones.
const PROMPT_MIN: usize = 4;

/// Rows of the smaller prompt pass, for the short prompts and prompt ends
/// that would leave most of a full pass's rows idle: every kernel of a
/// pass runs its full rows, those past its tokens returning at once.
const PROMPT_SHORT: usize = 64;

/// The device and its kernels.
struct Gpu {
    device: Rc<Device>,
    kernels: Kernels,
}

impl Gpu {
    fn matrix(&self, m: &Matrix) -> Result<DeviceMatrix, LoadError> {
        DeviceMatrix::upload(&self.device, m).map_err(vulkan_error)
    }

    fn vector(&self, v: &[f32]) -> Result<DeviceBuffer<f32>, LoadError> {
        DeviceBuffer::from_slice(&self.device, v).map_err(vulkan_error)
    }
}

fn vulkan_error(e: kinetic_vulkan::Error) -> LoadError {
    match e {
        kinetic_vulkan::Error::Unsupported(ty) => {
            let api = kinetic_vulkan::API;
            LoadError(format!(
                "{api} has no kernel for {ty}, a weight type this model uses; it runs on \
                 the CPU (--device cpu), and K-quants such as Q4_K_M have {api} kernels"
            ))
        }
        e => LoadError(format!("Vulkan: {e}")),
    }
}

/// The Vulkan devices that can run models, discrete GPUs first.
pub fn devices() -> kinetic_vulkan::Result<Vec<PhysicalDevice>> {
    let mut devices = Instance::new()?.physical_devices()?;
    devices.retain(|d| d.info().missing().is_none());
    Ok(devices)
}

/// A model whose weights live on a Vulkan device.
///
/// Its own methods report device failures as errors. Driven through
/// [`Target`](crate::speculative::Target), which has no error path, a
/// device failure panics.
pub struct VulkanModel<'m> {
    gpu: Gpu,
    name: String,
    /// The driver's name, version string and version number.
    driver: String,
    /// Most tokens a state holds.
    context: usize,
    n_embd: usize,
    eps: f32,
    embd: Matrix<'m>,
    embd_scale: Option<f32>,
    output: DeviceMatrix,
    /// The output head as the model file holds it, for drafts.
    output_rows: Matrix<'m>,
    output_norm: DeviceBuffer<f32>,
    layers: Vec<VulkanLayer>,
    mtp: Option<VulkanMtp>,
    ropes: RopeTables,
    /// Most tokens a prompt pass runs: what the device's kernels ask for,
    /// or [`MAX_BATCH`] if it has none of its own for such passes.
    prompt_batch: usize,
    /// Whether prompt passes run on the matrix units, which give other
    /// bits than the kernels tokens are decoded with.
    matrix: bool,
}

/// A sequence's caches and buffers on the device.
pub struct VulkanState {
    pos: usize,
    checkpoint: Option<usize>,
    /// Whether the tokens being run are a prompt's; see
    /// [`VulkanModel::set_prompt`].
    prompt: bool,
    decode: Pass,
    layers: Vec<LayerState>,
    /// Prompt passes, which the device may run with kernels written for
    /// many tokens at once, by rows, fewest first; none where it has no
    /// such kernels. Each has buffers of its own: a pass costs a little by
    /// the rows its buffers hold, which would slow every token decoded.
    prompt_passes: Vec<Pass>,
    /// What a rollback runs, when a layer needs it.
    replay: Option<Graph>,
    /// Host copies of recurrent layers at earlier lengths, for the warm
    /// cache; see [`Model::mark`].
    marks: Vec<Mark>,
}

impl VulkanState {
    /// Number of tokens processed so far.
    pub fn len(&self) -> usize {
        self.pos
    }

    pub fn is_empty(&self) -> bool {
        self.pos == 0
    }
}

/// A pass's buffers and recordings for its possible outputs.
struct Pass {
    work: Work,
    state: Graph,
    hidden: Graph,
    last_logits: Graph,
    greedy: Graph,
    /// Downloads the rows after the first, indexed by head row count minus two.
    logits_tail: Vec<Option<Graph>>,
    logits: Graph,
}

impl Pass {
    fn launch_logits(
        &mut self,
        device: &Rc<Device>,
        vocab: usize,
        rows: usize,
    ) -> kinetic_vulkan::Result<()> {
        if rows == 1 {
            return self.logits.launch();
        }
        let tail = &mut self.logits_tail[rows - 2];
        let copy = match tail {
            Some(copy) => copy,
            None => {
                let mut commands = Commands::new(device)?;
                commands.download(
                    &self.work.logits,
                    vocab,
                    &self.work.logits_host,
                    vocab,
                    (rows - 1) * vocab,
                )?;
                tail.insert(commands.finish()?)
            }
        };
        self.logits.launch_then(copy)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Head {
    None,
    Greedy,
    Logits,
}

/// What a run produces besides the state it advances.
struct Output {
    /// The greedy token after each row given to the head.
    greedy: Vec<u32>,
    /// The logit of each greedy token.
    top: Vec<f32>,
    /// The normalized final hidden state of every token.
    hidden: Vec<f32>,
    /// Logits of each row given to the head, when asked for.
    logits: Vec<f32>,
}

impl<'m> VulkanModel<'m> {
    /// Uploads `model` to the first Vulkan device that can run it, for
    /// sequences of up to `context` tokens. With `matrix`, prompt passes run
    /// on the device's matrix units where it has them; see
    /// [`VulkanModel::matrix`].
    pub fn load(
        model: &Model<'m>,
        context: usize,
        matrix: bool,
    ) -> Result<VulkanModel<'m>, LoadError> {
        let physical = devices()
            .map_err(vulkan_error)?
            .into_iter()
            .next()
            .ok_or_else(|| LoadError("no Vulkan device can run models".into()))?;
        VulkanModel::load_on(model, context, &physical, matrix)
    }

    /// Uploads `model` to `physical` for sequences of up to `context` tokens;
    /// `matrix` as for [`VulkanModel::load`].
    pub fn load_on(
        model: &Model<'m>,
        context: usize,
        physical: &PhysicalDevice,
        matrix: bool,
    ) -> Result<VulkanModel<'m>, LoadError> {
        if model.folded {
            return Err(LoadError(
                "weights stored folded are not supported on Vulkan yet".into(),
            ));
        }
        if model.per_layer.is_some() || model.logit_cap.is_some() {
            return Err(LoadError(
                "per-layer inputs and capped logits are not supported on Vulkan yet".into(),
            ));
        }
        let e = &model.embd;
        dequantize_row(e.ty, e.data, 0, &mut vec![0.0; e.cols])
            .map_err(|_| LoadError(format!("{} embeddings are not supported", e.ty)))?;
        let device = Device::new(physical).map_err(vulkan_error)?;
        let gpu = Gpu {
            kernels: Kernels::load_with(&device, matrix).map_err(vulkan_error)?,
            device,
        };
        let prompt_batch = gpu.kernels.prompt_batch().max(MAX_BATCH);
        let mut ropes = RopeTables::new(prompt_batch);
        let layers = model
            .layers
            .iter()
            .map(|l| VulkanLayer::upload(&gpu, l, &mut ropes))
            .collect::<Result<Vec<_>, _>>()?;
        let mtp = model
            .mtp
            .as_ref()
            .map(|m| VulkanMtp::upload(&gpu, m, &mut ropes))
            .transpose()?;
        Ok(VulkanModel {
            name: physical.info().name.clone(),
            driver: {
                let info = physical.info();
                format!(
                    "{} {} ({:#x})",
                    info.driver_name, info.driver_info, info.driver_version
                )
            },
            context,
            n_embd: model.n_embd,
            eps: model.eps,
            embd: model.embd,
            embd_scale: model.embd_scale,
            output: gpu.matrix(&model.output)?,
            output_rows: model.output,
            mtp,
            output_norm: gpu.vector(&model.output_norm)?,
            layers,
            ropes,
            prompt_batch,
            matrix: gpu.kernels.matrix(),
            gpu,
        })
    }

    /// Whether prompt passes run on the matrix units. Their products are
    /// sums of f16 terms, close to the decode kernels' but not equal, so a
    /// prompt's state differs slightly from the same tokens decoded one at
    /// a time. Decode steps and draft checks never run there, so drafts
    /// still give exactly the tokens plain decoding would; and every token
    /// marked as a prompt's runs there, so where a prompt is cut changes no
    /// bit.
    pub fn matrix(&self) -> bool {
        self.matrix
    }

    /// Turns [`VulkanModel::matrix`] off, along with the prompt passes it
    /// takes, for states made from now on: they need far less memory.
    pub fn without_matrix(&mut self) {
        if self.matrix {
            self.matrix = false;
            self.prompt_batch = MAX_BATCH;
        }
    }

    /// Marks the tokens `state` runs from now on as a prompt's (`true`) or
    /// as decoded ones. With [`VulkanModel::matrix`], prompt tokens all run
    /// in prompt passes, however few, as long as no checkpoint is set.
    pub fn set_prompt(&self, state: &mut VulkanState, prompt: bool) {
        state.prompt = prompt;
    }

    /// Most tokens the model can process in one prompt pass.
    pub fn prefill_batch(&self) -> usize {
        self.prompt_batch
    }

    /// The name of the device the model runs on.
    pub fn device_name(&self) -> &str {
        &self.name
    }

    /// The device's driver: its name, version string and version number.
    pub fn driver(&self) -> &str {
        &self.driver
    }

    pub fn n_vocab(&self) -> usize {
        self.output.rows()
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    /// Device bytes taken by the weights.
    pub fn weight_bytes(&self) -> usize {
        let layers: usize = self.layers.iter().map(VulkanLayer::size_bytes).sum();
        let mtp = self.mtp.as_ref().map_or(0, VulkanMtp::size_bytes);
        layers + mtp + self.output.size_bytes()
    }

    /// An empty state for a new sequence.
    pub fn new_state(&self) -> kinetic_vulkan::Result<VulkanState> {
        let dev = &self.gpu.device;
        let new_work = |rows, head_rows| {
            let batch = Batch::new(dev, rows, head_rows, self.context, self.ropes.words())?;
            Work::new(&self.gpu, &self.layers, self.n_embd, batch, self.n_vocab())
        };
        let work = new_work(MAX_BATCH, MAX_BATCH)?;
        let layers = self
            .layers
            .iter()
            .map(|l| l.new_state(dev, self.context, self.prompt_batch))
            .collect::<kinetic_vulkan::Result<Vec<_>>>()?;
        let replay = match self.layers.iter().any(VulkanLayer::replays) {
            true => Some(self.record_replay(&work, &layers)?),
            false => None,
        };
        // A prompt pass wants the logits of its last token at most.
        let mut sizes = vec![self.prompt_batch];
        if self.prompt_batch > PROMPT_SHORT {
            sizes.insert(0, PROMPT_SHORT);
        }
        let prompt_passes = match self.prompt_batch > MAX_BATCH {
            true => sizes
                .into_iter()
                .map(|rows| self.record_pass(new_work(rows, 1)?, &layers, true))
                .collect::<kinetic_vulkan::Result<Vec<_>>>()?,
            false => Vec::new(),
        };
        Ok(VulkanState {
            pos: 0,
            checkpoint: None,
            prompt: false,
            decode: self.record_pass(work, &layers, false)?,
            prompt_passes,
            replay,
            layers,
            marks: Vec::new(),
        })
    }

    /// Empties `state` for a new sequence, keeping its buffers.
    pub fn reset(&self, state: &mut VulkanState) -> kinetic_vulkan::Result<()> {
        state.pos = 0;
        state.checkpoint = None;
        state.prompt = false;
        state.marks.clear();
        for (layer, s) in self.layers.iter().zip(&mut state.layers) {
            layer.clear(s)?;
        }
        Ok(())
    }

    /// Device bytes held by `state`.
    pub fn state_bytes(&self, state: &VulkanState) -> usize {
        let layers: usize = state.layers.iter().map(LayerState::size_bytes).sum();
        let prompt: usize = state
            .prompt_passes
            .iter()
            .map(|p| p.work.size_bytes())
            .sum();
        state.decode.work.size_bytes() + prompt + layers
    }

    fn record_pass(
        &self,
        work: Work,
        states: &[LayerState],
        prompt: bool,
    ) -> kinetic_vulkan::Result<Pass> {
        let last = self.layers.len() - 1;
        let complete_layer = |c: &mut Commands| {
            c.set_prompt(prompt);
            self.layers[last].record_output(
                &self.gpu,
                &work,
                &states[last],
                &self.output_norm,
                self.eps,
                c,
            )
        };
        let mut hidden = Commands::new(&self.gpu.device)?;
        complete_layer(&mut hidden)?;
        hidden.download(
            work.h.values(),
            0,
            &work.hidden,
            0,
            work.batch.rows() * self.n_embd,
        )?;
        let mut last_logits = Commands::new(&self.gpu.device)?;
        complete_layer(&mut last_logits)?;
        self.record_head(&work, Head::Logits, &mut last_logits)?;
        last_logits.download(&work.logits, 0, &work.logits_host, 0, self.n_vocab())?;
        Ok(Pass {
            state: self.record(&work, states, Head::None, prompt)?,
            hidden: hidden.finish()?,
            last_logits: last_logits.finish()?,
            greedy: self.record(&work, states, Head::Greedy, prompt)?,
            logits: self.record(&work, states, Head::Logits, prompt)?,
            logits_tail: (1..work.batch.head_rows()).map(|_| None).collect(),
            work,
        })
    }

    /// Records the layers, the selected head output and the readback.
    fn record(
        &self,
        w: &Work,
        states: &[LayerState],
        head: Head,
        prompt: bool,
    ) -> kinetic_vulkan::Result<Graph> {
        let (k, d) = (&self.gpu.kernels, self.n_embd);
        let mut c = Commands::new(&self.gpu.device)?;
        c.set_prompt(prompt);
        let rows = w.batch.rows();
        c.independent(|c| {
            w.batch.record_upload(c)?;
            c.upload(&w.embedded, 0, &w.x, 0, rows * d)
        })?;
        let first = &self.layers[0].mixer_norm;
        k.norm(&w.batch, &w.x, None, first, self.eps, &w.h, &mut c)?;
        for (i, (layer, state)) in self.layers.iter().zip(states).enumerate() {
            let next = self.layers.get(i + 1);
            if head == Head::None && next.is_none() {
                layer.record_state(&self.gpu, w, state, &mut c)?;
            } else {
                let norm = next.map_or(&self.output_norm, |l| &l.mixer_norm);
                layer.record(&self.gpu, w, state, norm, self.eps, &mut c)?;
            }
        }
        if head == Head::None {
            return c.finish();
        }
        self.record_head(w, head, &mut c)?;
        let vocab = self.n_vocab();
        c.independent(|c| {
            c.download(w.h.values(), 0, &w.hidden, 0, rows * d)?;
            match head {
                Head::None => Ok(()),
                Head::Greedy => c.download(&w.best, 0, &w.greedy, 0, w.greedy.len()),
                Head::Logits => c.download(&w.logits, 0, &w.logits_host, 0, vocab),
            }
        })?;
        c.finish()
    }

    fn record_head(&self, w: &Work, head: Head, c: &mut Commands) -> kinetic_vulkan::Result<()> {
        let k = &self.gpu.kernels;
        let vocab = self.n_vocab();
        let out = Out {
            stride: vocab,
            offset: 0,
        };
        k.gemv_rows(&self.output, &w.h, &w.logits, out, Rows::Head, &w.batch, c)?;
        if head == Head::Greedy {
            k.argmax(&w.batch, &w.logits, vocab, &w.best, c)?;
        }
        Ok(())
    }

    /// Records what a rollback runs: every layer's replay of the tokens it
    /// keeps.
    fn record_replay(&self, w: &Work, states: &[LayerState]) -> kinetic_vulkan::Result<Graph> {
        let mut c = Commands::new(&self.gpu.device)?;
        w.batch.record_upload(&mut c)?;
        for (layer, state) in self.layers.iter().zip(states) {
            layer.record_replay(&self.gpu, w, state, &mut c)?;
        }
        c.finish()
    }

    /// Runs `tokens` in order. With `all_logits` the logits after every
    /// token are returned, `tokens.len()` rows of [`VulkanModel::n_vocab`]
    /// values; otherwise only those after the last one.
    pub fn run(
        &self,
        state: &mut VulkanState,
        tokens: &[u32],
        all_logits: bool,
    ) -> kinetic_vulkan::Result<Forward> {
        let out = self.forward(state, tokens, all_logits, Head::Logits, true)?;
        Ok(Forward {
            logits: out.logits,
            hidden: out.hidden,
        })
    }

    /// Runs `tokens` like [`VulkanModel::run`] but returns only the greedy
    /// token after every row (with `all_rows`) or after the last, and the
    /// normalized final hidden states; logits never leave the device.
    pub fn run_greedy(
        &self,
        state: &mut VulkanState,
        tokens: &[u32],
        all_rows: bool,
    ) -> kinetic_vulkan::Result<(Vec<u32>, Vec<f32>)> {
        let out = self.forward(state, tokens, all_rows, Head::Greedy, true)?;
        Ok((out.greedy, out.hidden))
    }

    /// [`VulkanModel::run`] for a choice that needs only the largest logit
    /// of each row: rows hold that logit at its greedy token and negative
    /// infinity elsewhere, so only a token and a float per row leave the
    /// device.
    pub fn run_top1(
        &self,
        state: &mut VulkanState,
        tokens: &[u32],
        all_rows: bool,
    ) -> kinetic_vulkan::Result<Forward> {
        let out = self.forward(state, tokens, all_rows, Head::Greedy, true)?;
        let vocab = self.n_vocab();
        let mut logits = vec![f32::NEG_INFINITY; out.greedy.len() * vocab];
        for (row, (&t, &v)) in out.greedy.iter().zip(&out.top).enumerate() {
            logits[row * vocab + t as usize] = v;
        }
        Ok(Forward {
            logits,
            hidden: out.hidden,
        })
    }

    /// Advances the sequence and returns the normalized hidden state of every token.
    pub fn advance(
        &self,
        state: &mut VulkanState,
        tokens: &[u32],
    ) -> kinetic_vulkan::Result<Vec<f32>> {
        Ok(self.forward(state, tokens, false, Head::None, true)?.hidden)
    }

    /// Advances the sequence, leaving its outputs on the device.
    pub fn advance_state(
        &self,
        state: &mut VulkanState,
        tokens: &[u32],
    ) -> kinetic_vulkan::Result<()> {
        self.forward(state, tokens, false, Head::None, false)
            .map(|_| ())
    }

    fn last_logits(
        &self,
        state: &mut VulkanState,
        tokens: &[u32],
    ) -> kinetic_vulkan::Result<Vec<f32>> {
        Ok(self
            .forward(state, tokens, false, Head::Logits, false)?
            .logits)
    }

    /// Marks the current length so a later [`VulkanModel::rollback`] can
    /// return to any point after it. At most [`MAX_BATCH`] tokens may run
    /// between the two.
    pub fn checkpoint(&self, state: &mut VulkanState) -> kinetic_vulkan::Result<()> {
        if let Some(start) = state.checkpoint {
            self.rollback(state, state.pos - start)?;
        }
        state.checkpoint = Some(state.pos);
        Ok(())
    }

    /// Keeps the first `keep` tokens run since the checkpoint and discards
    /// the rest, leaving the state exactly as if they had never run.
    pub fn rollback(&self, state: &mut VulkanState, keep: usize) -> kinetic_vulkan::Result<()> {
        let start = state
            .checkpoint
            .take()
            .expect("rollback without a checkpoint");
        assert!(
            start + keep <= state.pos,
            "cannot keep more tokens than were run"
        );
        if let Some(replay) = state.replay.as_ref().filter(|_| keep > 0) {
            state.decode.work.batch.set_replay(start, start + keep)?;
            replay.launch()?;
        }
        state.pos = start + keep;
        Ok(())
    }

    fn check(&self, state: &VulkanState, tokens: &[u32]) -> kinetic_vulkan::Result<()> {
        let vocab = self.n_vocab();
        let since = state.checkpoint.map(|start| state.pos - start);
        let msg = if tokens.is_empty() {
            "empty batch".to_string()
        } else if let Some(t) = tokens.iter().find(|&&t| t as usize >= vocab) {
            format!("token {t} is not below {vocab}")
        } else if state.pos + tokens.len() > self.context {
            format!(
                "{} tokens past a context of {}",
                state.pos + tokens.len(),
                self.context
            )
        } else if since.is_some_and(|s| s + tokens.len() > MAX_BATCH) {
            format!("more than {MAX_BATCH} tokens between a checkpoint and its rollback")
        } else {
            return Ok(());
        };
        Err(kinetic_vulkan::Error::Shape(msg))
    }

    fn forward(
        &self,
        state: &mut VulkanState,
        tokens: &[u32],
        all_rows: bool,
        output: Head,
        read_hidden: bool,
    ) -> kinetic_vulkan::Result<Output> {
        self.check(state, tokens)?;
        let (d, vocab) = (self.n_embd, self.n_vocab());
        let mut out = Output {
            greedy: Vec::new(),
            top: Vec::new(),
            hidden: Vec::with_capacity(if read_hidden { tokens.len() * d } else { 0 }),
            logits: Vec::new(),
        };
        let mut rest = tokens;
        while !rest.is_empty() {
            // Tokens whose rows are not each wanted, and that no draft may
            // be rolled back over, are prompt processing, which takes them
            // in passes as large as it can. Prompt passes on the matrix
            // units give their own bits, so there every token of a prompt
            // takes one, and only those; a prompt whose every row is wanted
            // takes one pass per token, since a prompt pass keeps the head
            // rows of its last token only.
            let prompt = self.prompt_batch > MAX_BATCH
                && state.checkpoint.is_none()
                && match self.matrix {
                    true => state.prompt,
                    false => !all_rows && rest.len() >= PROMPT_MIN,
                };
            let n = rest.len().min(match (prompt, all_rows) {
                (true, true) => 1,
                (true, false) => self.prompt_batch,
                (false, _) => MAX_BATCH,
            });
            let chunk;
            (chunk, rest) = rest.split_at(n);
            let head: Range<usize> = match (output, all_rows, rest.is_empty()) {
                (Head::None, _, _) | (_, false, false) => n..n,
                (_, true, _) => 0..n,
                (_, false, true) => n - 1..n,
            };
            // the smallest prompt pass that holds them
            let pass = prompt
                .then(|| {
                    state
                        .prompt_passes
                        .iter_mut()
                        .find(|p| p.work.batch.rows() >= n)
                })
                .flatten();
            let pass = pass.unwrap_or(&mut state.decode);
            let w = &mut pass.work;
            w.batch.set(n, state.pos, head.clone(), state.checkpoint)?;
            self.ropes.fill(&mut w.batch, state.pos, n);
            let x = &mut w.embedded.as_mut_slice()[..n * d];
            embed_into(&self.embd, self.embd_scale, chunk, x);
            match (head.len(), output, read_hidden) {
                (0, _, false) => pass.state.launch()?,
                (0, _, true) => pass.state.launch_then(&pass.hidden)?,
                (1, Head::Logits, false) => pass.state.launch_then(&pass.last_logits)?,
                (rows, Head::Logits, _) => pass.launch_logits(&self.gpu.device, vocab, rows)?,
                _ => pass.greedy.launch()?,
            }

            let w = &pass.work;
            if output == Head::Greedy {
                for pair in w.greedy.as_slice()[..2 * head.len()].chunks_exact(2) {
                    out.greedy.push(pair[0]);
                    out.top.push(f32::from_bits(pair[1]));
                }
            }
            if read_hidden {
                out.hidden.extend_from_slice(&w.hidden.as_slice()[..n * d]);
            }
            if output == Head::Logits {
                let rows = &w.logits_host.as_slice()[..head.len() * vocab];
                out.logits.extend_from_slice(rows);
            }
            state.pos += n;
        }
        Ok(out)
    }
}
