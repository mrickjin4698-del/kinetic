//! Prompts on a Vulkan GPU, on a real model: where a prompt is cut into
//! passes changes no bit of its logits, and asking for every row's logits
//! gives each row the bits of the prompt that ends there. On a GPU with
//! cooperative matrices this covers prompt passes on the matrix units.
//!
//! KINETIC_TEST_GGUF=model.gguf cargo test --release -p kinetic-models --features vulkan --test vulkan -- --ignored

#![cfg(feature = "vulkan")]

use kinetic_gguf::GgufFile;
use kinetic_models::vulkan::{VulkanModel, VulkanState};

/// Longer than a prompt pass of 512 tokens, to cut across one.
const LEN: usize = 600;

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// A prompt unlike any training data: `LEN` tokens spread over the vocabulary.
fn prompt(vocab: usize) -> Vec<u32> {
    let mut x = 12345u64;
    (0..LEN)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as usize % (vocab - 2) + 1) as u32
        })
        .collect()
}

/// The logits after `tokens`, run as a prompt in pieces of `cut`.
fn run_cut(gpu: &VulkanModel, state: &mut VulkanState, tokens: &[u32], cut: usize) -> Vec<u32> {
    gpu.reset(state).unwrap();
    gpu.set_prompt(state, true);
    let mut last = Vec::new();
    for piece in tokens.chunks(cut) {
        last = gpu.run(state, piece, false).unwrap().logits;
    }
    bits(&last)
}

/// One test, so one model on the device.
#[test]
#[ignore = "needs KINETIC_TEST_GGUF and a Vulkan device"]
fn prompts_give_the_same_bits_however_cut() {
    let path = std::env::var("KINETIC_TEST_GGUF").expect("set KINETIC_TEST_GGUF");
    let file = GgufFile::open(&path).unwrap();
    let model = kinetic_models::load(&file).unwrap();
    let gpu = VulkanModel::load(&model, LEN + 8, true).unwrap();
    eprintln!("{}, matrix units: {}", gpu.device_name(), gpu.matrix());
    let ids = prompt(gpu.n_vocab());
    let mut state = gpu.new_state().unwrap();

    let whole = run_cut(&gpu, &mut state, &ids, ids.len());
    for cut in [1, 3, 17, 64, 65, 100, 256, 511, 512, 513] {
        assert_eq!(run_cut(&gpu, &mut state, &ids, cut), whole, "cut {cut}");
    }

    // The hidden-only prefix still feeds the drafter and the next scored pass.
    let cut = 513;
    gpu.reset(&mut state).unwrap();
    gpu.set_prompt(&mut state, true);
    let scored = gpu.run(&mut state, &ids[..cut], false).unwrap().hidden;
    gpu.reset(&mut state).unwrap();
    gpu.set_prompt(&mut state, true);
    let advanced = gpu.advance(&mut state, &ids[..cut]).unwrap();
    assert_eq!(bits(&advanced), bits(&scored));
    assert_eq!(state.len(), cut);
    assert_eq!(
        bits(&gpu.run(&mut state, &ids[cut..], false).unwrap().logits),
        whole
    );

    // every row, as /logits asks for them
    let n = 80;
    gpu.reset(&mut state).unwrap();
    gpu.set_prompt(&mut state, true);
    let rows = gpu.run(&mut state, &ids[..n], true).unwrap().logits;
    let vocab = gpu.n_vocab();
    assert_eq!(rows.len(), n * vocab);
    for end in [1, 2, 33, 64, 65, n] {
        assert_eq!(
            bits(&rows[(end - 1) * vocab..end * vocab]),
            run_cut(&gpu, &mut state, &ids[..end], end),
            "row {end}"
        );
    }
}
