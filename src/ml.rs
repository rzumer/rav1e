// Copyright (c) 2016-2018, Alliance for Open Media. All rights reserved
// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

/// Neural network configuration
#[derive(Clone)]
pub struct NeuralNetworkConfig {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub num_hidden_layers: usize,
    pub num_hidden_nodes: [usize; NeuralNetworkConfig::MAX_HIDDEN_LAYERS],
    pub weights: [[f32; NeuralNetworkConfig::MAX_CONNECTIONS_PER_LAYER]; NeuralNetworkConfig::MAX_HIDDEN_LAYERS + 1],
    pub bias: [[f32; NeuralNetworkConfig::MAX_NODES_PER_LAYER]; NeuralNetworkConfig::MAX_HIDDEN_LAYERS + 1]
}

impl NeuralNetworkConfig {
    pub const MAX_HIDDEN_LAYERS: usize = 10;
    pub const MAX_NODES_PER_LAYER: usize = 128;
    pub const MAX_CONNECTIONS_PER_LAYER: usize = Self::MAX_NODES_PER_LAYER * Self::MAX_NODES_PER_LAYER;
}

pub fn nn_predict(input: &Vec<f32>, nn_config: &NeuralNetworkConfig, output: &mut Vec<f32>) {
    let mut num_input_nodes = nn_config.num_inputs;
    let mut buffer = [[0_f32; NeuralNetworkConfig::MAX_NODES_PER_LAYER]; 2];
    let mut buffer_index = 0_usize;

    // Copy input vector to buffer since borrows collide
    for i in 0_usize..num_input_nodes {
        buffer[1][i] = input[i];
    }

    // Propagate hidden layers
    let num_layers = nn_config.num_hidden_layers;
    assert!(num_layers <= NeuralNetworkConfig::MAX_HIDDEN_LAYERS);

    for layer in 0..num_layers {
        let bias = nn_config.bias[layer];
        let num_output_nodes = nn_config.num_hidden_nodes[layer] as usize;
        assert!(num_output_nodes < NeuralNetworkConfig::MAX_NODES_PER_LAYER);

        for node_out in 0_usize..num_output_nodes {
            let weights = &nn_config.weights[layer][node_out * num_input_nodes..];
            let mut val = 0_f32;

            for node_in in 0_usize..num_input_nodes {
                val += weights[node_in] * buffer[1_usize - buffer_index][node_in];
            }

            val += bias[node_out];

            // ReLU as activation function
            val = val.max(0_f32);

            buffer[buffer_index][node_out] = val;
        }

        num_input_nodes = num_output_nodes;
        buffer_index = 1_usize - buffer_index;
    }

    // Final output layer
    for node_out in 0_usize..nn_config.num_outputs {
        let weights = &nn_config.weights[num_layers][node_out * num_input_nodes..];
        let bias = nn_config.bias[num_layers];
        let mut val = 0_f32;

        for node_in in 0_usize..num_input_nodes {
            val += weights[node_in] * buffer[1 - buffer_index][node_in];
        }

        output[node_out] = val + bias[node_out];
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use libc;
    use rand::{ChaChaRng, Rng};
    use std;

    const MAX_ITER: usize = 50000;

    extern {
        #[cfg(test)]
        fn av1_nn_predict(features: *const libc::c_float, nn_config: *const NN_CONFIG,
                          output: *mut libc::c_float);
    }

    // Copied from C
    #[repr(C)]
    struct NN_CONFIG {
        num_inputs: libc::c_int,         // Number of input nodes, i.e. features.
        num_outputs: libc::c_int,        // Number of output nodes.
        num_hidden_layers: libc::c_int,  // Number of hidden layers, maximum 10.
        // Number of nodes for each hidden layer.
        num_hidden_nodes: [libc::c_int; NeuralNetworkConfig::MAX_HIDDEN_LAYERS],
        // Weight parameters, indexed by layer.
        weights: [*const libc::c_float; NeuralNetworkConfig::MAX_HIDDEN_LAYERS + 1],
        // Bias parameters, indexed by layer.
        bias: [*const libc::c_float; NeuralNetworkConfig::MAX_HIDDEN_LAYERS + 1]
    }

    impl NN_CONFIG {
        fn new(config: &NeuralNetworkConfig) -> Self {
            let mut num_hidden_nodes: [libc::c_int; NeuralNetworkConfig::MAX_HIDDEN_LAYERS] =
                unsafe { std::mem::uninitialized() };
            let mut weights: [*const libc::c_float; NeuralNetworkConfig::MAX_HIDDEN_LAYERS + 1] =
                unsafe { std::mem::uninitialized() };
            let mut bias: [*const libc::c_float; NeuralNetworkConfig::MAX_HIDDEN_LAYERS + 1] =
                unsafe { std::mem::uninitialized() };

            for i in 0_usize..NeuralNetworkConfig::MAX_HIDDEN_LAYERS {
                num_hidden_nodes[i] = config.num_hidden_nodes[i] as libc::c_int;
                weights[i] = config.weights[i].as_ptr();
                bias[i] = config.bias[i].as_ptr();
            }

            NN_CONFIG {
                num_inputs: config.num_inputs as libc::c_int,
                num_outputs: config.num_outputs as libc::c_int,
                num_hidden_layers: config.num_hidden_layers as libc::c_int,
                num_hidden_nodes: num_hidden_nodes,
                weights: weights,
                bias: bias
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

    fn pred_aom(input: &Vec<f32>, nn_config: &NeuralNetworkConfig, output: &mut Vec<f32>) {
        // Remap manually, since map-collect does not work on arrays
        let config = NN_CONFIG::new(&nn_config);

        unsafe {
            av1_nn_predict(input.as_ptr(), &config, output.as_mut_ptr() )
        }
    }

    fn do_pred(ra: &mut ChaChaRng) -> (Vec<f32>, Vec<f32>) {
        let (input, mut o1, mut o2) = setup_pred(ra);

        let mut weights =
            [[0_f32; NeuralNetworkConfig::MAX_CONNECTIONS_PER_LAYER]; NeuralNetworkConfig::MAX_HIDDEN_LAYERS + 1];
        let mut bias =
            [[0_f32; NeuralNetworkConfig::MAX_NODES_PER_LAYER]; NeuralNetworkConfig::MAX_HIDDEN_LAYERS + 1];

        for i in 0..4 {
            for j in 0..4 {
                weights[i][j] = i as f32 + j as f32;
                bias[i][j] = i as f32 - j as f32;
            }
        }

        let config = NeuralNetworkConfig {
            num_inputs: 32,
            num_outputs: 32,
            num_hidden_layers: 4,
            num_hidden_nodes: [4, 4, 4, 4, 0, 0, 0, 0, 0, 0],
            weights: weights,
            bias: bias
        };

        pred_aom(&input, &config.clone(), &mut o1);
        nn_predict(&input, &config.clone(), &mut o2);

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