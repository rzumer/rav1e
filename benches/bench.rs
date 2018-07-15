// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#[macro_use]
extern crate bencher;
extern crate libc;
extern crate rand;
extern crate rav1e;

use bencher::*;
use rand::{ChaChaRng, Rng};
use rav1e::context::*;
use rav1e::ec;
use rav1e::ml::*;
use rav1e::partition::*;
use rav1e::predict::*;
use rav1e::*;

// Copied from C
// FIXME: duplicated from ml.rs
#[repr(C)]
struct NN_CONFIG {
  num_inputs: libc::c_int, // Number of input nodes, i.e. features.
  num_outputs: libc::c_int, // Number of output nodes.
  num_hidden_layers: libc::c_int, // Number of hidden layers, maximum 10.
  // Number of nodes for each hidden layer.
  num_hidden_nodes: [libc::c_int; NeuralNetwork::MAX_HIDDEN_LAYERS],
  // Weight parameters, indexed by layer.
  weights: [*const libc::c_float; NeuralNetwork::MAX_HIDDEN_LAYERS + 1],
  // Bias parameters, indexed by layer.
  bias: [*const libc::c_float; NeuralNetwork::MAX_HIDDEN_LAYERS + 1],
}

impl NN_CONFIG {
  fn new(config: &NeuralNetwork) -> Self {
    let mut num_hidden_nodes =
      [0 as libc::c_int; NeuralNetwork::MAX_HIDDEN_LAYERS];
    let mut weights = [std::ptr::null::<libc::c_float>();
      NeuralNetwork::MAX_HIDDEN_LAYERS + 1];
    let mut biases = [std::ptr::null::<libc::c_float>();
      NeuralNetwork::MAX_HIDDEN_LAYERS + 1];

    for i in 0_usize..NeuralNetwork::MAX_HIDDEN_LAYERS {
      num_hidden_nodes[i] = config.num_hidden_nodes[i] as libc::c_int;
      weights[i] = config.weights[i].as_ptr();
      biases[i] = config.biases[i].as_ptr();
    }

    NN_CONFIG {
      num_inputs: config.num_inputs as libc::c_int,
      num_outputs: config.num_outputs as libc::c_int,
      num_hidden_layers: config.num_hidden_layers as libc::c_int,
      num_hidden_nodes: num_hidden_nodes,
      weights: weights,
      bias: biases,
    }
  }
}

extern {
  fn highbd_dc_predictor(
    dst: *mut u16,
    stride: libc::ptrdiff_t,
    bw: libc::c_int,
    bh: libc::c_int,
    above: *const u16,
    left: *const u16,
    bd: libc::c_int,
  );

  fn highbd_h_predictor(
    dst: *mut u16,
    stride: libc::ptrdiff_t,
    bw: libc::c_int,
    bh: libc::c_int,
    above: *const u16,
    left: *const u16,
    bd: libc::c_int,
  );

  fn highbd_v_predictor(
    dst: *mut u16,
    stride: libc::ptrdiff_t,
    bw: libc::c_int,
    bh: libc::c_int,
    above: *const u16,
    left: *const u16,
    bd: libc::c_int,
  );

  fn highbd_paeth_predictor(
    dst: *mut u16,
    stride: libc::ptrdiff_t,
    bw: libc::c_int,
    bh: libc::c_int,
    above: *const u16,
    left: *const u16,
    bd: libc::c_int,
  );

  fn highbd_smooth_predictor(
    dst: *mut u16,
    stride: libc::ptrdiff_t,
    bw: libc::c_int,
    bh: libc::c_int,
    above: *const u16,
    left: *const u16,
    bd: libc::c_int,
  );

  fn highbd_smooth_h_predictor(
    dst: *mut u16,
    stride: libc::ptrdiff_t,
    bw: libc::c_int,
    bh: libc::c_int,
    above: *const u16,
    left: *const u16,
    bd: libc::c_int,
  );

  fn highbd_smooth_v_predictor(
    dst: *mut u16,
    stride: libc::ptrdiff_t,
    bw: libc::c_int,
    bh: libc::c_int,
    above: *const u16,
    left: *const u16,
    bd: libc::c_int,
  );

  fn av1_nn_predict(
    features: *const libc::c_float,
    nn_config: *const NN_CONFIG,
    output: *mut libc::c_float,
  );
}

#[inline(always)]
fn pred_dc_4x4(
  output: &mut [u16],
  stride: usize,
  above: &[u16],
  left: &[u16],
) {
  unsafe {
    highbd_dc_predictor(
      output.as_mut_ptr(),
      stride as libc::ptrdiff_t,
      4,
      4,
      above.as_ptr(),
      left.as_ptr(),
      8,
    );
  }
}

#[inline(always)]
fn pred_h_4x4(output: &mut [u16], stride: usize, above: &[u16], left: &[u16]) {
  unsafe {
    highbd_h_predictor(
      output.as_mut_ptr(),
      stride as libc::ptrdiff_t,
      4,
      4,
      above.as_ptr(),
      left.as_ptr(),
      8,
    );
  }
}

#[inline(always)]
fn pred_v_4x4(output: &mut [u16], stride: usize, above: &[u16], left: &[u16]) {
  unsafe {
    highbd_v_predictor(
      output.as_mut_ptr(),
      stride as libc::ptrdiff_t,
      4,
      4,
      above.as_ptr(),
      left.as_ptr(),
      8,
    );
  }
}

#[inline(always)]
fn pred_paeth_4x4(
  output: &mut [u16],
  stride: usize,
  above: &[u16],
  left: &[u16],
) {
  unsafe {
    highbd_paeth_predictor(
      output.as_mut_ptr(),
      stride as libc::ptrdiff_t,
      4,
      4,
      above.as_ptr(),
      left.as_ptr(),
      8,
    );
  }
}

#[inline(always)]
fn pred_smooth_4x4(
  output: &mut [u16],
  stride: usize,
  above: &[u16],
  left: &[u16],
) {
  unsafe {
    highbd_smooth_predictor(
      output.as_mut_ptr(),
      stride as libc::ptrdiff_t,
      4,
      4,
      above.as_ptr(),
      left.as_ptr(),
      8,
    );
  }
}

#[inline(always)]
fn pred_smooth_h_4x4(
  output: &mut [u16],
  stride: usize,
  above: &[u16],
  left: &[u16],
) {
  unsafe {
    highbd_smooth_h_predictor(
      output.as_mut_ptr(),
      stride as libc::ptrdiff_t,
      4,
      4,
      above.as_ptr(),
      left.as_ptr(),
      8,
    );
  }
}

#[inline(always)]
fn pred_smooth_v_4x4(
  output: &mut [u16],
  stride: usize,
  above: &[u16],
  left: &[u16],
) {
  unsafe {
    highbd_smooth_v_predictor(
      output.as_mut_ptr(),
      stride as libc::ptrdiff_t,
      4,
      4,
      above.as_ptr(),
      left.as_ptr(),
      8,
    );
  }
}

#[inline(always)]
fn pred_nn(input: &[f32], config: &NN_CONFIG, output: &mut [f32]) {
  unsafe { av1_nn_predict(input.as_ptr(), config, output.as_mut_ptr()) }
}

const MAX_ITER: usize = 50000;

fn setup_pred(ra: &mut ChaChaRng) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let output = vec![0u16; 32 * 32];
  let above: Vec<u16> = (0..32).map(|_| ra.gen()).collect();
  let left: Vec<u16> = (0..32).map(|_| ra.gen()).collect();

  (above, left, output)
}

fn setup_pred_nn(
  ra: &mut ChaChaRng,
) -> (Vec<f32>, ml::NeuralNetwork, Vec<f32>) {
  static mut WEIGHTS: [[f32; ml::NeuralNetwork::MAX_WEIGHTS_PER_LAYER];
    ml::NeuralNetwork::MAX_HIDDEN_LAYERS + 1] = [[0_f32;
    ml::NeuralNetwork::MAX_WEIGHTS_PER_LAYER];
    ml::NeuralNetwork::MAX_HIDDEN_LAYERS + 1];

  for i in 0..ml::NeuralNetwork::MAX_HIDDEN_LAYERS + 1 {
    for j in 0..ml::NeuralNetwork::MAX_WEIGHTS_PER_LAYER {
      unsafe {
        WEIGHTS[i][j] = ra.gen();
      }
    }
  }

  static mut BIASES: [[f32; ml::NeuralNetwork::MAX_NODES_PER_LAYER];
    ml::NeuralNetwork::MAX_HIDDEN_LAYERS + 1] = [[0_f32;
    ml::NeuralNetwork::MAX_NODES_PER_LAYER];
    ml::NeuralNetwork::MAX_HIDDEN_LAYERS + 1];

  for i in 0..ml::NeuralNetwork::MAX_HIDDEN_LAYERS + 1 {
    for j in 0..ml::NeuralNetwork::MAX_NODES_PER_LAYER {
      unsafe {
        BIASES[i][j] = ra.gen();
      }
    }
  }

  let config = NeuralNetwork {
    num_inputs: 14,
    num_outputs: 4,
    num_hidden_layers: 4,
    num_hidden_nodes: &[12, 8, 6, 4, 0, 0, 0, 0, 0, 0],
    weights: unsafe { &WEIGHTS },
    biases: unsafe { &BIASES },
  };

  let output = vec![0f32; config.num_outputs];
  let input: Vec<f32> = (0..config.num_inputs).map(|_| ra.gen()).collect();

  (input, config, output)
}

fn intra_dc_pred_native(b: &mut Bencher) {
  let mut ra = ChaChaRng::new_unseeded();
  let (above, left, mut output) = setup_pred(&mut ra);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      Block4x4::pred_dc(&mut output, 32, &above[..4], &left[..4]);
    }
  })
}

fn intra_dc_pred_aom(b: &mut Bencher) {
  let mut ra = ChaChaRng::new_unseeded();
  let (above, left, mut output) = setup_pred(&mut ra);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      pred_dc_4x4(&mut output, 32, &above[..4], &left[..4]);
    }
  })
}

fn intra_h_pred_native(b: &mut Bencher) {
  let mut ra = ChaChaRng::new_unseeded();
  let (_above, left, mut output) = setup_pred(&mut ra);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      Block4x4::pred_h(&mut output, 32, &left[..4]);
    }
  })
}

fn intra_h_pred_aom(b: &mut Bencher) {
  let mut ra = ChaChaRng::new_unseeded();
  let (above, left, mut output) = setup_pred(&mut ra);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      pred_h_4x4(&mut output, 32, &above[..4], &left[..4]);
    }
  })
}

fn intra_v_pred_native(b: &mut Bencher) {
  let mut ra = ChaChaRng::new_unseeded();
  let (above, _left, mut output) = setup_pred(&mut ra);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      Block4x4::pred_v(&mut output, 32, &above[..4]);
    }
  })
}

fn intra_v_pred_aom(b: &mut Bencher) {
  let mut ra = ChaChaRng::new_unseeded();
  let (above, left, mut output) = setup_pred(&mut ra);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      pred_v_4x4(&mut output, 32, &above[..4], &left[..4]);
    }
  })
}

fn intra_paeth_pred_native(b: &mut Bencher) {
  let mut ra = ChaChaRng::new_unseeded();
  let (above, left, mut output) = setup_pred(&mut ra);
  let above_left = unsafe { *above.as_ptr().offset(-1) };

  b.iter(|| {
    for _ in 0..MAX_ITER {
      Block4x4::pred_paeth(
        &mut output,
        32,
        &above[..4],
        &left[..4],
        above_left,
      );
    }
  })
}

fn intra_paeth_pred_aom(b: &mut Bencher) {
  let mut ra = ChaChaRng::new_unseeded();
  let (above, left, mut output) = setup_pred(&mut ra);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      pred_paeth_4x4(&mut output, 32, &above[..4], &left[..4]);
    }
  })
}

fn intra_smooth_pred_native(b: &mut Bencher) {
  let mut ra = ChaChaRng::new_unseeded();
  let (above, left, mut output) = setup_pred(&mut ra);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      Block4x4::pred_smooth(&mut output, 32, &above[..4], &left[..4], 8);
    }
  })
}

fn intra_smooth_pred_aom(b: &mut Bencher) {
  let mut ra = ChaChaRng::new_unseeded();
  let (above, left, mut output) = setup_pred(&mut ra);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      pred_smooth_4x4(&mut output, 32, &above[..4], &left[..4]);
    }
  })
}

fn intra_smooth_h_pred_native(b: &mut Bencher) {
  let mut ra = ChaChaRng::new_unseeded();
  let (above, left, mut output) = setup_pred(&mut ra);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      Block4x4::pred_smooth_h(&mut output, 32, &above[..4], &left[..4], 8);
    }
  })
}

fn intra_smooth_h_pred_aom(b: &mut Bencher) {
  let mut ra = ChaChaRng::new_unseeded();
  let (above, left, mut output) = setup_pred(&mut ra);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      pred_smooth_h_4x4(&mut output, 32, &above[..4], &left[..4]);
    }
  })
}

fn intra_smooth_v_pred_native(b: &mut Bencher) {
  let mut ra = ChaChaRng::new_unseeded();
  let (above, left, mut output) = setup_pred(&mut ra);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      Block4x4::pred_smooth_v(&mut output, 32, &above[..4], &left[..4], 8);
    }
  })
}

fn intra_smooth_v_pred_aom(b: &mut Bencher) {
  let mut ra = ChaChaRng::new_unseeded();
  let (above, left, mut output) = setup_pred(&mut ra);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      pred_smooth_v_4x4(&mut output, 32, &above[..4], &left[..4]);
    }
  })
}

fn nn_pred_native(b: &mut Bencher) {
  let mut ra = ChaChaRng::new_unseeded();

  let (input, config, _) = setup_pred_nn(&mut ra);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      config.predict(&input);
    }
  })
}

fn nn_pred_aom(b: &mut Bencher) {
  let mut ra = ChaChaRng::new_unseeded();

  let (input, config, mut output) = setup_pred_nn(&mut ra);
  let c_config = NN_CONFIG::new(&config);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      pred_nn(&input, &c_config, &mut output);
    }
  })
}

struct WriteB {
  tx_size: TxSize,
  qi: usize,
}

impl TDynBenchFn for WriteB {
  fn run(&self, b: &mut Bencher) {
    write_b_bench(b, self.tx_size, self.qi);
  }
}

pub fn write_b() -> Vec<TestDescAndFn> {
  use std::borrow::Cow;
  let mut benches = ::std::vec::Vec::new();
  for &tx_size in &[TxSize::TX_4X4, TxSize::TX_8X8] {
    for &qi in &[20, 55] {
      let w = WriteB { tx_size, qi };
      let n = format!("write_b_bench({:?}, {})", tx_size, qi);
      benches.push(TestDescAndFn {
        desc: TestDesc {
          name: Cow::from(n),
          ignore: false,
        },
        testfn: TestFn::DynBenchFn(Box::new(w)),
      });
    }
  }
  benches
}

fn write_b_bench(b: &mut Bencher, tx_size: TxSize, qindex: usize) {
  unsafe {
    av1_rtcd();
    aom_dsp_rtcd();
  }
  let mut fi = FrameInvariants::new(1024, 1024, qindex, 10);
  let w = ec::Writer::new();
  let fc = CDFContext::new(fi.qindex as u8);
  let bc = BlockContext::new(fi.sb_width * 16, fi.sb_height * 16);
  let mut fs = FrameState::new(&fi);
  let mut cw = ContextWriter::new(w, fc, bc);

  let tx_type = TxType::DCT_DCT;

  let sbx = 0;
  let sby = 0;

  b.iter(|| {
    for &mode in RAV1E_INTRA_MODES {
      let sbo = SuperBlockOffset { x: sbx, y: sby };
      for p in 1..3 {
        for by in 0..8 {
          for bx in 0..8 {
            let bo = sbo.block_offset(bx, by);
            let tx_bo = BlockOffset {
              x: bo.x + bx,
              y: bo.y + by,
            };
            let po = tx_bo.plane_offset(&fs.input.planes[p].cfg);
            encode_tx_block(
              &mut fi,
              &mut fs,
              &mut cw,
              p,
              &bo,
              mode,
              tx_size,
              tx_type,
              tx_size.block_size(),
              &po,
              false,
            );
          }
        }
      }
    }
  });
}

benchmark_group!(
  intra,
  intra_dc_pred_native,
  intra_dc_pred_aom,
  intra_h_pred_native,
  intra_h_pred_aom,
  intra_v_pred_native,
  intra_v_pred_aom,
  intra_paeth_pred_native,
  intra_paeth_pred_aom,
  intra_smooth_pred_native,
  intra_smooth_pred_aom,
  intra_smooth_h_pred_native,
  intra_smooth_h_pred_aom,
  intra_smooth_v_pred_native,
  intra_smooth_v_pred_aom
);

benchmark_group!(ml, nn_pred_native, nn_pred_aom);

benchmark_main!(intra, write_b, ml);
