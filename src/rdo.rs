// Copyright (c) 2001-2016, Alliance for Open Media. All rights reserved
// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_camel_case_types)]
#![cfg_attr(feature = "cargo-clippy", allow(cast_lossless))]

use context::*;
use ec::OD_BITRES;
use encode_block;
use ml::nn_models::*;
use partition::*;
use plane::*;
use predict::{RAV1E_INTRA_MODES, RAV1E_INTRA_MODES_MINIMAL};
use quantize::dc_q;
use std;
use std::f64;
use std::vec::Vec;
use write_tx_blocks;
use BlockSize;
use FrameInvariants;
use FrameState;
use FrameType;
use Tune;

#[derive(Clone)]
pub struct RDOOutput {
  pub rd_cost: f64,
  pub part_type: PartitionType,
  pub part_modes: Vec<RDOPartitionOutput>,
}

#[derive(Clone)]
pub struct RDOPartitionOutput {
  pub rd_cost: f64,
  pub bo: BlockOffset,
  pub pred_mode_luma: PredictionMode,
  pub pred_mode_chroma: PredictionMode,
  pub skip: bool,
}

#[allow(unused)]
fn cdef_dist_wxh_8x8(src1: &PlaneSlice, src2: &PlaneSlice) -> u64 {
  //TODO: Handle high bit-depth here by setting coeff_shift
  let coeff_shift = 0;
  let mut sum_s: i32 = 0;
  let mut sum_d: i32 = 0;
  let mut sum_s2: i64 = 0;
  let mut sum_d2: i64 = 0;
  let mut sum_sd: i64 = 0;
  for j in 0..8 {
    for i in 0..8 {
      let s = src1.p(i, j) as i32;
      let d = src2.p(i, j) as i32;
      sum_s += s;
      sum_d += d;
      sum_s2 += (s * s) as i64;
      sum_d2 += (d * d) as i64;
      sum_sd += (s * d) as i64;
    }
  }
  let svar = (sum_s2 - ((sum_s as i64 * sum_s as i64 + 32) >> 6)) as f64;
  let dvar = (sum_d2 - ((sum_d as i64 * sum_d as i64 + 32) >> 6)) as f64;
  let sse = (sum_d2 + sum_s2 - 2 * sum_sd) as f64;
  //The two constants were tuned for CDEF, but can probably be better tuned for use in general RDO
  let ssim_boost = 0.5_f64 * (svar + dvar + (400 << 2 * coeff_shift) as f64)
    / f64::sqrt((20000 << 4 * coeff_shift) as f64 + svar * dvar);
  (sse * ssim_boost + 0.5_f64) as u64
}

#[allow(unused)]
fn cdef_dist_wxh(
  src1: &PlaneSlice, src2: &PlaneSlice, w: usize, h: usize
) -> u64 {
  let mut sum: u64 = 0;
  for j in 0..h / 8 {
    for i in 0..w / 8 {
      sum += cdef_dist_wxh_8x8(
        &src1.subslice(i * 8, j * 8),
        &src2.subslice(i * 8, j * 8)
      )
    }
  }
  sum
}

// Sum of Squared Error for a wxh block
fn sse_wxh(src1: &PlaneSlice, src2: &PlaneSlice, w: usize, h: usize) -> u64 {
  let mut sse: u64 = 0;

  for j in 0..h {
    for i in 0..w {
      let dist = (src1.p(i, j) as i16 - src2.p(i, j) as i16) as i64;
      sse += (dist * dist) as u64;
    }
  }

  sse
}

// Normalized SSE
fn sse_norm_wxh(
  src1: &PlaneSlice,
  src2: &PlaneSlice,
  w: usize,
  h: usize,
) -> f32 {
  (sse_wxh(src1, src2, w, h) as f32) / ((w * h) as f32)
}

// Mean error
fn mean_error_wxh(
  src1: &PlaneSlice,
  src2: &PlaneSlice,
  w: usize,
  h: usize,
) -> f32 {
  let mut error: u64 = 0;

  for j in 0..h {
    for i in 0..w {
      error += (src1.p(i, j) as i16 - src2.p(i, j) as i16).abs() as u64;
    }
  }

  (error as f32) / ((w * h) as f32)
}

// Compute the rate-distortion cost for an encode
fn compute_rd_cost(
  fi: &FrameInvariants,
  fs: &FrameState,
  w_y: usize,
  h_y: usize,
  w_uv: usize,
  h_uv: usize,
  partition_start_x: usize,
  partition_start_y: usize,
  bo: &BlockOffset,
  bit_cost: u32,
) -> f64 {
  let q = dc_q(fi.config.quantizer) as f64;

  // Convert q into Q0 precision, given that libaom quantizers are Q3
  let q0 = q / 8.0_f64;

  // Lambda formula from doc/theoretical_results.lyx in the daala repo
  // Use Q0 quantizer since lambda will be applied to Q0 pixel domain
  let lambda = q0 * q0 * std::f64::consts::LN_2 / 6.0;

  // Compute distortion
  let po = bo.plane_offset(&fs.input.planes[0].cfg);
  let mut distortion = if fi.config.tune == Tune::Psnr {
    sse_wxh(
      &fs.input.planes[0].slice(&po),
      &fs.rec.planes[0].slice(&po),
      w_y,
      h_y
    )
  } else if fi.config.tune == Tune::Psychovisual {
    cdef_dist_wxh(
      &fs.input.planes[0].slice(&po),
      &fs.rec.planes[0].slice(&po),
      w_y,
      h_y
    )
  } else {
    unimplemented!();
  };

  // Add chroma distortion only when it is available
  if w_uv > 0 && h_uv > 0 {
    for p in 1..3 {
      let sb_offset = bo.sb_offset().plane_offset(&fs.input.planes[p].cfg);
      let po = PlaneOffset {
        x: sb_offset.x + partition_start_x,
        y: sb_offset.y + partition_start_y,
      };

      distortion += sse_wxh(
        &fs.input.planes[p].slice(&po),
        &fs.rec.planes[p].slice(&po),
        w_uv,
        h_uv,
      );
    }
  };

  // Compute rate
  let rate = (bit_cost as f64) / ((1 << OD_BITRES) as f64);

  (distortion as f64) + lambda * rate
}

// RDO-based mode decision
pub fn rdo_mode_decision(
  fi: &FrameInvariants,
  fs: &mut FrameState,
  cw: &mut ContextWriter,
  bsize: BlockSize,
  bo: &BlockOffset,
) -> RDOOutput {
  let mut best_mode_luma = PredictionMode::DC_PRED;
  let mut best_mode_chroma = PredictionMode::DC_PRED;
  let mut best_skip = false;
  let mut best_rd = std::f64::MAX;
  let tell = cw.w.tell_frac();

  // Get block luma and chroma dimensions
  let w = bsize.width();
  let h = bsize.height();

  let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;

  let mut w_uv = w >> xdec;
  let mut h_uv = h >> ydec;

  let is_chroma_block = has_chroma(bo, bsize, xdec, ydec);

  if (w_uv == 0 || h_uv == 0) && is_chroma_block {
    w_uv = 4;
    h_uv = 4;
  }

  let partition_start_x = (bo.x & LOCAL_BLOCK_MASK) >> xdec << MI_SIZE_LOG2;
  let partition_start_y = (bo.y & LOCAL_BLOCK_MASK) >> ydec << MI_SIZE_LOG2;

  let skip = false;

  let checkpoint = cw.checkpoint();

  // Exclude complex prediction modes at higher speed levels
  let mode_set = if fi.config.speed <= 3 {
    RAV1E_INTRA_MODES
  } else {
    RAV1E_INTRA_MODES_MINIMAL
  };

  for &luma_mode in mode_set {
    if fi.frame_type == FrameType::KEY
      && luma_mode >= PredictionMode::NEARESTMV
    {
      break;
    }

    if fi.use_nn_prediction {
      encode_block(fi, fs, cw, luma_mode, luma_mode, bsize, bo, true);

      let rd = 0f64;//model_rd_with_dnn(fi, fs, bsize, bo, 0);

      if rd < best_rd {
        best_rd = rd;
        best_mode_luma = luma_mode;
        best_mode_chroma = luma_mode;
        best_skip = skip;
      }

      cw.rollback(&checkpoint);
    } else if is_chroma_block && fi.config.speed <= 3 {
      // Find the best chroma prediction mode for the current luma prediction mode
      for &chroma_mode in RAV1E_INTRA_MODES {
        encode_block(fi, fs, cw, luma_mode, chroma_mode, bsize, bo, skip);

        let cost = cw.w.tell_frac() - tell;
        let rd = compute_rd_cost(fi, fs, w, h, w_uv, h_uv, 
          partition_start_x, partition_start_y, bo, cost);
          
        if rd < best_rd {
            best_rd = rd;
            best_mode_luma = luma_mode;
            best_mode_chroma = chroma_mode;
            best_skip = skip;
        }

        cw.rollback(&checkpoint);
      }
    } else {
      encode_block(fi, fs, cw, luma_mode, luma_mode, bsize, bo, skip);

      let cost = cw.w.tell_frac() - tell;
      let rd = compute_rd_cost(fi, fs, w, h, w_uv, h_uv, 
        partition_start_x, partition_start_y, bo, cost);

      if rd < best_rd {
        best_rd = rd;
        best_mode_luma = luma_mode;
        best_mode_chroma = luma_mode;
        best_skip = skip;
      }

      cw.rollback(&checkpoint);
    }
  }

  assert!(best_rd >= 0_f64);

  RDOOutput {
    rd_cost: best_rd,
    part_type: PartitionType::PARTITION_NONE,
    part_modes: vec![RDOPartitionOutput {
      bo: bo.clone(),
      pred_mode_luma: best_mode_luma,
      pred_mode_chroma: best_mode_chroma,
      rd_cost: best_rd,
      skip: best_skip,
    }],
  }
}

// RDO-based intra frame transform type decision
pub fn rdo_tx_type_decision(
  fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
  mode: PredictionMode, bsize: BlockSize, bo: &BlockOffset, tx_size: TxSize,
  tx_set: TxSet
) -> TxType {
  let mut best_type = TxType::DCT_DCT;
  let mut best_rd = std::f64::MAX;
  let tell = cw.w.tell_frac();

  // Get block luma and chroma dimensions
  let w = bsize.width();
  let h = bsize.height();

  let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;

  let mut w_uv = w >> xdec;
  let mut h_uv = h >> ydec;

  if (w_uv == 0 || h_uv == 0) && has_chroma(bo, bsize, xdec, ydec) {
    w_uv = 4;
    h_uv = 4;
  }

  let partition_start_x = (bo.x & LOCAL_BLOCK_MASK) >> xdec << MI_SIZE_LOG2;
  let partition_start_y = (bo.y & LOCAL_BLOCK_MASK) >> ydec << MI_SIZE_LOG2;

  let checkpoint = cw.checkpoint();

  for &tx_type in RAV1E_TX_TYPES {
    // Skip unsupported transform types
    if av1_tx_used[tx_set as usize][tx_type as usize] == 0 {
      continue;
    }

    write_tx_blocks(
      fi, fs, cw, mode, mode, bo, bsize, tx_size, tx_type, false,
    );

    let cost = cw.w.tell_frac() - tell;
    let rd = compute_rd_cost(
      fi,
      fs,
      w,
      h,
      w_uv,
      h_uv,
      partition_start_x,
      partition_start_y,
      bo,
      cost,
    );

    if rd < best_rd {
      best_rd = rd;
      best_type = tx_type;
    }

    cw.rollback(&checkpoint);
  }

  assert!(best_rd >= 0_f64);

  best_type
}

// RDO-based single level partitioning decision
pub fn rdo_partition_decision(
  fi: &FrameInvariants,
  fs: &mut FrameState,
  cw: &mut ContextWriter,
  bsize: BlockSize,
  bo: &BlockOffset,
  cached_block: &RDOOutput,
) -> RDOOutput {
  model_rd_with_dnn(fi, fs, bsize, bo, 0);

  let max_rd = std::f64::MAX;

  let mut best_partition = cached_block.part_type;
  let mut best_rd = cached_block.rd_cost;
  let mut best_pred_modes = cached_block.part_modes.clone();

  let checkpoint = cw.checkpoint();

  for &partition in RAV1E_PARTITION_TYPES {
    // Do not re-encode results we already have
    if partition == cached_block.part_type && cached_block.rd_cost < max_rd {
      continue;
    }

    let mut rd: f64;
    let mut child_modes = std::vec::Vec::new();

    match partition {
      PartitionType::PARTITION_NONE => {
        if bsize > BlockSize::BLOCK_32X32 {
          continue;
        }

        let mode_decision = cached_block
          .part_modes
          .get(0)
          .unwrap_or(&rdo_mode_decision(fi, fs, cw, bsize, bo).part_modes[0])
          .clone();
        child_modes.push(mode_decision);
      }
      PartitionType::PARTITION_SPLIT => {
        let subsize = get_subsize(bsize, partition);

        if subsize == BlockSize::BLOCK_INVALID {
          continue;
        }

        let bs = bsize.width_mi();
        let hbs = bs >> 1; // Half the block size in blocks

        let offset = BlockOffset { x: bo.x, y: bo.y };
        let mode_decision = rdo_mode_decision(fi, fs, cw, subsize, &offset)
          .part_modes[0]
          .clone();
        child_modes.push(mode_decision);

        let offset = BlockOffset { x: bo.x + hbs as usize, y: bo.y };
        let mode_decision = rdo_mode_decision(fi, fs, cw, subsize, &offset)
          .part_modes[0]
          .clone();
        child_modes.push(mode_decision);

        let offset = BlockOffset { x: bo.x, y: bo.y + hbs as usize };
        let mode_decision = rdo_mode_decision(fi, fs, cw, subsize, &offset)
          .part_modes[0]
          .clone();
        child_modes.push(mode_decision);

        let offset =
          BlockOffset { x: bo.x + hbs as usize, y: bo.y + hbs as usize };
        let mode_decision = rdo_mode_decision(fi, fs, cw, subsize, &offset)
          .part_modes[0]
          .clone();
        child_modes.push(mode_decision);
      }
      _ => {
        assert!(false);
      }
    }

    rd = child_modes.iter().map(|m| m.rd_cost).sum::<f64>();

    if rd < best_rd {
      best_rd = rd;
      best_partition = partition;
      best_pred_modes = child_modes.clone();
    }

    cw.rollback(&checkpoint);
  }

  assert!(best_rd >= 0_f64);

  RDOOutput {
    rd_cost: best_rd,
    part_type: best_partition,
    part_modes: best_pred_modes,
  }
}

// TODO: clean up and comment
fn model_rd_with_dnn(
  fi: &FrameInvariants,
  fs: &FrameState,
  bsize: BlockSize,
  bo: &BlockOffset,
  plane: usize,
) -> f64 {
  // TODO: fix chroma implementation
  if plane > 0 {
    unimplemented!();
  }

  let (mut rate, mut distortion) = (0f64, 0f64);

  let PlaneConfig { xdec, ydec, .. } = fs.input.planes[plane].cfg;

  let plane_bsize = get_plane_block_size(bsize, xdec, ydec);
  let po = bo.plane_offset(&fs.input.planes[plane].cfg);

  let log_numpels = num_pels_log2_lookup[plane_bsize as usize];
  let num_samples = 1 << log_numpels;

  let dequant_shift = 3; // bit depth - 5
  let q = dc_q(fi.config.quantizer);
  let q_step = q >> dequant_shift; // may be valid only for luma

  let shift = 0; // bit depth - 8

  let (bw, bh) = (bsize.width(), bsize.height());

  let sse = sse_wxh(
    &fs.input.planes[plane].slice(&po),
    &fs.rec.planes[plane].slice(&po),
    bw,
    bh,
  ) as f32;

  if sse > 0f32 {
    let sse_norm = sse / num_samples as f32;
    let mut sse_norm_arr = get_2x2_normalized_sses(fs, bsize, bo, plane);
    let mut mean = mean_error_wxh(
      &fs.input.planes[plane].slice(&po),
      &fs.rec.planes[plane].slice(&po),
      bw,
      bh,
    );

    if shift > 0 {
      for k in 0..4 {
        sse_norm_arr[k] /= (1 << (2 * shift)) as f32;
      }

      mean /= (1 << shift) as f32;
    }

    let sse_norm_sum = sse_norm_arr.iter().sum::<f32>();
    let mut sse_frac_arr: [f32; 3];

    if sse_norm_sum == 0f32 {
      sse_frac_arr = [0.25; 3];
    } else {
      sse_frac_arr = [0f32; 3];

      for k in 0..3 {
        sse_frac_arr[k] = sse_norm_arr[k] / sse_norm_sum;
      }
    }

    let q_sqr = (q_step * q_step) as f32;
    let q_sqr_by_sse_norm = q_sqr / (sse_norm + 1f32);
    let mean_sqr_by_sse_norm = mean * mean / (sse_norm + 1f32);
    let (hor_corr, vert_corr) = get_horver_correlation(
      &fs.input.planes[plane].slice(&po),
      &fs.rec.planes[plane].slice(&po),
      bw,
      bh,
    );

    let features: Vec<f32> = vec![
      hor_corr,
      log_numpels as f32,
      mean_sqr_by_sse_norm,
      q_sqr_by_sse_norm,
      sse_frac_arr[0],
      sse_frac_arr[1],
      sse_frac_arr[2],
      vert_corr
    ];

    for i in 0..8 {
      println!("Feature #{}: {}", i, features[i]);
    }

    let dist_by_sse_norm_f = DISTORTION_MODEL.predict(&features)[0];
    let rate_f = RATE_MODEL.predict(&features)[0] as f64;
    let dist_f = dist_by_sse_norm_f as f64 * (1f64 + sse_norm as f64);

    rate = ((rate_f * (1 << log_numpels) as f64) + 0.5).abs();
    distortion = ((dist_f * (1 << log_numpels) as f64) + 0.5).abs();

    // TODO: check if skip is better
    /*if (RDCOST(x->rdmult, rate_i, dist_i) >= RDCOST(x->rdmult, 0, (sse << 4))) {
      distortion = sse << 4;
      rate = 0;
    } else if (rate == 0) {
      distortion = sse << 4;
    }*/
  }
  
  println!("{}: R/D/E: {}/{}/{}", plane, rate, distortion, sse);

  let lambda = ((q * q) as f64) * std::f64::consts::LN_2 / 6.0;
  distortion + lambda * rate
}

// TODO: clean up and comment
fn get_2x2_normalized_sses(
  fs: &FrameState,
  tx_bsize: BlockSize,
  bo: &BlockOffset,
  plane: usize,
) -> [f32; 2 * 2] {
  if plane > 0 {
    unimplemented!();
  }

  let mut sse_norm_arr = [0f32; 2 * 2];

  let half_width = tx_bsize.width() / 2;
  let half_height = tx_bsize.height() / 2;

  for row in 0..2 {
    for col in 0..2 {
      let offset = bo.plane_offset(&fs.input.planes[plane].cfg);
      let po = PlaneOffset {
        x: offset.x + col * half_width,
        y: offset.y + row * half_height,
      };

      sse_norm_arr[row * 2 + col] = sse_norm_wxh(
        &fs.input.planes[plane].slice(&po),
        &fs.rec.planes[plane].slice(&po),
        half_width,
        half_height,
      );
    }
  }

  sse_norm_arr
}

// TODO: clean up and comment
fn get_horver_correlation(
  src1: &PlaneSlice,
  src2: &PlaneSlice,
  w: usize,
  h: usize,
) -> (f32, f32) {
  let num = (h - 1) * (w - 1);
  assert!(num > 0);

  let (mut hcorr, mut vcorr) = (1f32, 1f32);

  let num_r = 1f32 / (num as f32);
  let (mut x_sum, mut y_sum, mut z_sum) = (0i64, 0i64, 0i64);
  let (mut x2_sum, mut y2_sum, mut z2_sum) = (0i64, 0i64, 0i64);
  let (mut xy_sum, mut xz_sum) = (0i64, 0i64);

  for i in 1..h {
    for j in 1..w {
      let x = (src1.p(i, j) as i16 - src2.p(i, j) as i16) as i64;
      let y = (src1.p(i, j - 1) as i16 - src2.p(i, j - 1) as i16) as i64;
      let z = (src1.p(i - 1, j) as i16 - src2.p(i - 1, j) as i16) as i64;

      x_sum += x;
      y_sum += y;
      z_sum += z;

      x2_sum += x * x;
      y2_sum += y * y;
      z2_sum += z * z;

      xy_sum += x * y;
      xz_sum += x * z;
    }
  }

  let x_var_n = (x2_sum - (x_sum * x_sum)) as f32 * num_r;
  let y_var_n = (y2_sum - (y_sum * y_sum)) as f32 * num_r;
  let z_var_n = (z2_sum - (z_sum * z_sum)) as f32 * num_r;
  let xy_var_n = (xy_sum - (x_sum * y_sum)) as f32 * num_r;
  let xz_var_n = (xz_sum - (x_sum * z_sum)) as f32 * num_r;

  if x_var_n > 0f32 {
    if y_var_n > 0f32 {
      hcorr = (xy_var_n / (x_var_n * y_var_n).sqrt()).max(0f32);
    }

    if z_var_n > 0f32 {
      vcorr = (xz_var_n / (x_var_n * z_var_n).sqrt()).max(0f32);
    }
  }

  (hcorr, vcorr)
}
