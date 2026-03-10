use ndarray::prelude::*;

use crate::transformer_block::{AttentionBlock, LayerNormalization, LinearLayer, silu, softmax};
pub mod transformer_block;

fn main() {
    let layer_norm1 = LayerNormalization {
        beta: Array1::zeros(4),
        gamma: Array1::zeros(4),
        epsilon: 1e-5,
    };
    let attention_block = AttentionBlock {
        dimension: 4,
        w_q: Array2::eye(4),
        w_k: Array2::eye(4),
        w_v: Array2::eye(4),
        w_o: Array2::eye(4),
        b_q: 0.0,
        b_k: 0.0,
        b_v: 0.0,
        b_o: array![0.0, 0.0, 0.0, 0.0],
        nb_heads: 2,
    };
    let layer_norm2 = LayerNormalization {
        beta: Array1::zeros(4),
        gamma: Array1::zeros(4),
        epsilon: 1e-5,
    };

    let input = array![[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]];
    // Attention Block
    // Normalization
    let out = layer_norm1.run(input.clone()).unwrap();
    // Linear projections and split into heads
    let (queries, keys, values) = attention_block.linear_projections_and_heads(out);
    // Parallel processing of heads
    let mut heads = vec![];
    for (q, (k, v)) in queries
        .into_iter()
        .zip(keys.into_iter().zip(values.into_iter()))
    {
        // Scaled dot products
        let score = attention_block.scaled_dot_product(q, k);
        // softmax
        let softmax_score = softmax(score);
        // Linear layer with values
        let out = softmax_score.dot(&v);
        heads.push(out);
    }
    // Output projection
    let attention_out = attention_block.output_projection(heads);

    // Residual Addition
    let post_attention_out = input + attention_out;

    // Feed-forward Block
    // Normalization
    let out = layer_norm2.run(post_attention_out.clone()).unwrap();
    // LineaLayer
    let mut out = LinearLayer {
        w: Array2::ones((4, 6)),
        b: Array1::ones(6),
    }
    .run(out);
    // Activation Layer
    out.mapv_inplace(|elem| silu(elem));
    // LineaLayer
    let out = LinearLayer {
        w: Array2::ones((4, 2)),
        b: Array1::ones(2),
    }
    .run(out);

    // Residual Addition
    let final_out = post_attention_out + out;
}
