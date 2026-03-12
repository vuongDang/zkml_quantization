use ndarray::prelude::*;

use crate::transformer_block::{
    ActivationFunction, AttentionLayer, LinearLayer, NormalizationLayer, TransformerBlock, silu,
    softmax,
};
pub mod transformer_block;

fn main() {
    let layer_norm1 = NormalizationLayer {
        beta: Array1::zeros(4),
        gamma: Array1::zeros(4),
        epsilon: 1e-5,
    };
    let attention_block = AttentionLayer {
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
    let layer_norm2 = NormalizationLayer {
        beta: Array1::zeros(4),
        gamma: Array1::zeros(4),
        epsilon: 1e-5,
    };

    let linear_1 = LinearLayer {
        w: Array2::ones((4, 6)),
        b: Array1::ones(6),
    };

    let linear_2 = LinearLayer {
        w: Array2::ones((4, 2)),
        b: Array1::ones(2),
    };
    let activation = ActivationFunction::GELU;

    let block = TransformerBlock::new(
        layer_norm1,
        attention_block,
        layer_norm2,
        linear_1,
        linear_2,
        activation,
    );
    let input = array![[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]];

    let _ = block.run(input);
}
