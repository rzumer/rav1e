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
             NeuralNetwork::MAX_HIDDEN_LAYERS + 1],
}

impl NeuralNetwork {
  pub const MAX_HIDDEN_LAYERS: usize = 10;
  pub const MAX_NODES_PER_LAYER: usize = 128;
  pub const MAX_WEIGHTS_PER_LAYER: usize =
    (Self::MAX_NODES_PER_LAYER - 1) * (Self::MAX_NODES_PER_LAYER - 1);

  pub fn predict(&self, input: &Vec<f32>, output: &mut Vec<f32>) {
    let mut num_input_nodes = self.num_inputs;
    let mut buffer = [[0_f32; NeuralNetwork::MAX_NODES_PER_LAYER]; 2];
    let mut buffer_index = 0_usize;

    // Copy input vector to buffer since borrows collide
    for i in 0_usize..num_input_nodes {
      buffer[1][i] = input[i];
    }

    // Propagate hidden layers
    let num_layers = self.num_hidden_layers;
    assert!(num_layers <= NeuralNetwork::MAX_HIDDEN_LAYERS);

    for layer in 0..num_layers {
      let biases = self.biases[layer];
      let num_output_nodes = self.num_hidden_nodes[layer] as usize;

      // Bias is counted as one node towards the maximum
      assert!(num_output_nodes < NeuralNetwork::MAX_NODES_PER_LAYER);

      for node_out in 0_usize..num_output_nodes {
        let weights = &self.weights[layer][node_out * num_input_nodes..];
        let mut val = 0_f32;

        for node_in in 0_usize..num_input_nodes {
          val += weights[node_in] * buffer[1_usize - buffer_index][node_in];
        }

        val += biases[node_out];

        // ReLU as activation function
        val = val.max(0_f32);

        buffer[buffer_index][node_out] = val;
      }

      num_input_nodes = num_output_nodes;
      buffer_index = 1_usize - buffer_index;
    }

    // Final output layer
    for node_out in 0_usize..self.num_outputs {
      let weights = &self.weights[num_layers][node_out * num_input_nodes..];
      let biases = self.biases[num_layers];
      let mut val = 0_f32;

      for node_in in 0_usize..num_input_nodes {
        val += weights[node_in] * buffer[1 - buffer_index][node_in];
      }

      output[node_out] = val + biases[node_out];
    }
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
      features: *const libc::c_float,
      nn_config: *const NN_CONFIG,
      output: *mut libc::c_float,
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

  fn setup_pred(ra: &mut ChaChaRng) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let size: usize = 32;
    let output = vec![0f32; size];
    let input: Vec<f32> = (0..size).map(|_| ra.gen()).collect();

    let o1 = output.clone();
    let o2 = output.clone();

    (input, o1, o2)
  }

  fn pred_aom(
    input: &Vec<f32>,
    nn_config: &NeuralNetwork,
    output: &mut Vec<f32>,
  ) {
    let config = NN_CONFIG::new(&nn_config);

    unsafe { av1_nn_predict(input.as_ptr(), &config, output.as_mut_ptr()) }
  }

  fn do_pred(ra: &mut ChaChaRng) -> (Vec<f32>, Vec<f32>) {
    let (input, mut o1, mut o2) = setup_pred(ra);

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
      num_inputs: 14,
      num_outputs: 4,
      num_hidden_layers: 4,
      num_hidden_nodes: &[12, 8, 6, 4, 0, 0, 0, 0, 0, 0],
      weights: unsafe { &WEIGHTS },
      biases: unsafe { &BIASES },
    };

    pred_aom(&input, &config.clone(), &mut o1);
    config.predict(&input, &mut o2);

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
