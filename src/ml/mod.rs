// Copyright (c) 2016-2018, Alliance for Open Media. All rights reserved
// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

pub mod nn_models;

/// Neural network configuration
#[derive(Clone)]
pub struct NeuralNetwork {
  pub num_inputs: usize,
  pub num_outputs: usize,
  pub num_hidden_layers: usize,
  pub num_hidden_nodes: &'static [usize; NeuralNetwork::MAX_HIDDEN_LAYERS],
  pub weights: &'static [[f32; NeuralNetwork::MAX_WEIGHTS_PER_LAYER];
             NeuralNetwork::MAX_HIDDEN_LAYERS + 1],
  pub biases: &'static [[f32; NeuralNetwork::MAX_NODES_PER_LAYER];
             NeuralNetwork::MAX_HIDDEN_LAYERS + 1]
}

impl NeuralNetwork {
  pub const MAX_HIDDEN_LAYERS: usize = 10;
  pub const MAX_NODES_PER_LAYER: usize = 128;
  pub const MAX_WEIGHTS_PER_LAYER: usize =
    (Self::MAX_NODES_PER_LAYER - 1) * (Self::MAX_NODES_PER_LAYER - 1);

  pub fn predict(&self, input: &Vec<f32>) -> Vec<f32> {
    let mut num_input_nodes = self.num_inputs;
    let mut buffer_0 = [0_f32; NeuralNetwork::MAX_NODES_PER_LAYER];
    let mut buffer_1 = [0_f32; NeuralNetwork::MAX_NODES_PER_LAYER];

    let mut buf_in = &mut buffer_0[..num_input_nodes];
    let mut buf_out = &mut buffer_1[..num_input_nodes];
    // Copy input vector to buffer since borrows collide
    buf_in.copy_from_slice(&input[..num_input_nodes]);

    // Propagate hidden layers
    let num_layers = self.num_hidden_layers;
    assert!(num_layers <= NeuralNetwork::MAX_HIDDEN_LAYERS);

    #[inline(always)]
    fn propagate(
      num_output_nodes: usize, num_input_nodes: usize, layer_weights: &[f32],
      layer_biases: &[f32], buf_in: &[f32], buf_out: &mut [f32],
      cb: fn(f32) -> f32
    ) {
      for node_out in 0_usize..num_output_nodes {
        let start = node_out * num_input_nodes;
        let end = start + num_input_nodes;
        let weights = &layer_weights[start..end];
        let bias = layer_biases[node_out];
        let val = bias + {
          weights
            .iter()
            .zip(buf_in.iter())
            .fold(0f32, |acc, (w, b)| acc + w * b)
        };

        buf_out[node_out] = cb(val);
      }
    }

    for ((biases, &num_output_nodes), layer_weights) in self.biases
      [..num_layers]
      .iter()
      .zip(self.num_hidden_nodes[..num_layers].iter())
      .zip(self.weights[..num_layers].iter())
    {
      // Bias is counted as one node towards the maximum
      assert!(num_output_nodes < NeuralNetwork::MAX_NODES_PER_LAYER);

      propagate(
        num_output_nodes,
        num_input_nodes,
        layer_weights,
        biases,
        buf_in,
        buf_out,
        |v: f32| -> f32 { v.max(0f32) }
      );

      use std::mem;

      num_input_nodes = num_output_nodes;
      mem::swap(&mut buf_in, &mut buf_out);
    }

    let mut output = vec![0f32; self.num_outputs];

    // Final output layer
    let biases = &self.biases[num_layers];
    let layer_weights = &self.weights[num_layers];

    propagate(
      self.num_outputs,
      num_input_nodes,
      layer_weights,
      biases,
      buf_in,
      &mut output,
      |v: f32| -> f32 { v }
    );

    output
  }
}

#[cfg(test)]
pub mod test {
  use super::*;
  use libc;
  use rand::{ChaChaRng, Rng};
  use std;

  const MAX_ITER: usize = 10;

  extern {
    #[cfg(test)]
    fn av1_nn_predict(
      features: *const libc::c_float, nn_config: *const NN_CONFIG,
      output: *mut libc::c_float
    );
  }

  // Copied from C
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
    bias: [*const libc::c_float; NeuralNetwork::MAX_HIDDEN_LAYERS + 1]
  }

  impl NN_CONFIG {
    fn new(config: &NeuralNetwork) -> Self {
      let mut num_hidden_nodes =
        [0 as libc::c_int; NeuralNetwork::MAX_HIDDEN_LAYERS];
      let mut weights = [std::ptr::null::<libc::c_float>();
        NeuralNetwork::MAX_HIDDEN_LAYERS + 1];
      let mut biases = [std::ptr::null::<libc::c_float>();
        NeuralNetwork::MAX_HIDDEN_LAYERS + 1];

      for i in 0_usize..NeuralNetwork::MAX_HIDDEN_LAYERS + 1 {
        if i < NeuralNetwork::MAX_HIDDEN_LAYERS {
          num_hidden_nodes[i] = config.num_hidden_nodes[i] as libc::c_int;
        }

        weights[i] = config.weights[i].as_ptr();
        biases[i] = config.biases[i].as_ptr();
      }

      NN_CONFIG {
        num_inputs: config.num_inputs as libc::c_int,
        num_outputs: config.num_outputs as libc::c_int,
        num_hidden_layers: config.num_hidden_layers as libc::c_int,
        num_hidden_nodes,
        weights,
        bias: biases
      }
    }
  }

  fn setup_pred(
    ra: &mut ChaChaRng, num_inputs: usize, num_outputs: usize
  ) -> (Vec<f32>, Vec<f32>) {
    let input: Vec<f32> = (0..num_inputs).map(|_| ra.gen()).collect();
    let output = vec![0f32; num_outputs];

    (input, output)
  }

  fn pred_aom(
    input: &Vec<f32>, nn_config: &NeuralNetwork, output: &mut Vec<f32>
  ) {
    let config = NN_CONFIG::new(&nn_config);

    unsafe { av1_nn_predict(input.as_ptr(), &config, output.as_mut_ptr()) }
  }

  fn do_pred(ra: &mut ChaChaRng) -> (Vec<f32>, Vec<f32>) {
    let num_inputs = 14;
    let num_outputs = 4;
    let (input, mut o1) = setup_pred(ra, num_inputs, num_outputs);

    // Declare test weights and biases as static, such that they do not overflow the stack during tests
    static mut WEIGHTS: [[f32; NeuralNetwork::MAX_WEIGHTS_PER_LAYER];
      NeuralNetwork::MAX_HIDDEN_LAYERS + 1] = [[0_f32;
      NeuralNetwork::MAX_WEIGHTS_PER_LAYER];
      NeuralNetwork::MAX_HIDDEN_LAYERS + 1];

    for i in 0..NeuralNetwork::MAX_HIDDEN_LAYERS + 1 {
      for j in 0..NeuralNetwork::MAX_WEIGHTS_PER_LAYER {
        unsafe {
          WEIGHTS[i][j] = ra.gen();
        }
      }
    }

    static mut BIASES: [[f32; NeuralNetwork::MAX_NODES_PER_LAYER];
      NeuralNetwork::MAX_HIDDEN_LAYERS + 1] = [[0_f32;
      NeuralNetwork::MAX_NODES_PER_LAYER];
      NeuralNetwork::MAX_HIDDEN_LAYERS + 1];

    for i in 0..NeuralNetwork::MAX_HIDDEN_LAYERS + 1 {
      for j in 0..NeuralNetwork::MAX_NODES_PER_LAYER {
        unsafe {
          BIASES[i][j] = ra.gen();
        }
      }
    }

    let config = NeuralNetwork {
      num_inputs,
      num_outputs,
      num_hidden_layers: 4,
      num_hidden_nodes: &[12, 8, 6, 4, 0, 0, 0, 0, 0, 0],
      weights: unsafe { &WEIGHTS },
      biases: unsafe { &BIASES }
    };

    pred_aom(&input, &config.clone(), &mut o1);
    let o2 = config.predict(&input);

    (o1, o2)
  }

  #[test]
  fn pred_matches() {
    let mut ra = ChaChaRng::new_unseeded();
    for _ in 0..MAX_ITER {
      let (o1, o2) = do_pred(&mut ra);
      assert_eq!(o1, o2);
    }
  }
}
