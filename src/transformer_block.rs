//! Steps in a transformer block
//!
//! Attention
//!     1. LayerNorm
//!     2. Linear projections → Q, K, V matrices
//!     3. Scaled dot-product → QKᵀ / √d_k
//!     4. Softmax → attention weights
//!     5. Weighted sum → softmax · V
//!     6. Output projection → concatenate heads · W_O
//! 7. Residual addition → x + attention_output
//! 8. LayerNorm (second one, before FFN)
//! Feed-Forward
//!     9. First linear layer → xW_1 + b_1
//!     10. Activation → GELU/SiLU/ReLU
//!     11. Second linear layer → xW_2 + b_2
//! 12. Residual addition → x + ffn_output

type Matrix = Array2<f32>;

use ndarray::{Array1, Array2, Axis, concatenate, s};
use std::{
    f32,
    ops::{Div, Sub},
};

pub struct TransformerBlock {
    normalization_1: NormalizationLayer,
    attention: AttentionLayer,
    normalization_2: NormalizationLayer,
    linear_1: LinearLayer,
    activation: ActivationFunction,
    linear_2: LinearLayer,
}

impl TransformerBlock {
    pub fn new(
        normalization_1: NormalizationLayer,
        attention: AttentionLayer,
        normalization_2: NormalizationLayer,
        linear_1: LinearLayer,
        linear_2: LinearLayer,
        activation: ActivationFunction,
    ) -> Self {
        TransformerBlock {
            normalization_1,
            attention,
            normalization_2,
            linear_1,
            activation,
            linear_2,
        }
    }

    pub fn run(&self, input: Matrix) -> Result<Matrix, TransformerError> {
        // Attention Block
        // Normalization
        let out = self.normalization_1.run(input.clone())?;
        // Linear projections and split into heads
        let (queries, keys, values) = self.attention.linear_projections_and_heads(out);
        // Parallel processing of heads
        let mut heads = vec![];
        for (q, (k, v)) in queries
            .into_iter()
            .zip(keys.into_iter().zip(values.into_iter()))
        {
            // Scaled dot products
            let score = self.attention.scaled_dot_product(q, k);
            // softmax
            let softmax_score = softmax(score);
            // Linear layer with values
            let out = softmax_score.dot(&v);
            heads.push(out);
        }
        // Output projection
        let attention_out = self.attention.output_projection(heads);

        // Residual Addition
        let post_attention_out = input + attention_out;

        // Feed-forward Block
        // Normalization
        let out = self.normalization_2.run(post_attention_out.clone())?;
        // LineaLayer
        let mut out = self.linear_1.run(out);
        // Activation Layer
        out.mapv_inplace(|elem| self.activation.run(elem));
        // LineaLayer
        let out = self.linear_2.run(out);

        // Residual Addition
        let final_out = post_attention_out + out;
        Ok(final_out)
    }
}

use thiserror::Error;

pub struct NormalizationLayer {
    pub beta: Array1<f32>,
    pub gamma: Array1<f32>,
    pub epsilon: f32,
}

impl NormalizationLayer {
    pub fn run(&self, input: Matrix) -> Result<Matrix, TransformerError> {
        let mut output = Matrix::zeros(input.raw_dim());
        for (i, row) in input.axis_iter(Axis(0)).enumerate() {
            let g_i = self
                .gamma
                .get(i)
                .ok_or(InvalidDimension("gamma is too small".to_string()))?;
            let b_i = self
                .beta
                .get(i)
                .ok_or(InvalidDimension("beta is too small".to_string()))?;
            let variance = row.var(0.0);
            let mean = row
                .mean()
                .ok_or(InvalidDimension("Row is empty".to_string()))?;
            let std_deviation = (variance + self.epsilon).sqrt();
            let res = ((row.sub(mean)).div(std_deviation)) * (*g_i) + (*b_i);
            output.row_mut(i).assign(&res);
        }

        Ok(output)
    }
}

// Inputs are sequence lengths 2 and dimensions 3
#[derive(Default)]
pub struct AttentionLayer {
    pub dimension: usize,
    pub nb_heads: usize,

    // Query, Key and Value weights and biases
    pub w_q: Matrix,
    pub b_q: f32,
    pub w_k: Matrix,
    pub b_k: f32,
    pub w_v: Matrix,
    pub b_v: f32,
    pub w_o: Matrix,
    pub b_o: Array1<f32>,
}

use TransformerError::*;
impl AttentionLayer {
    pub fn linear_projections_and_heads(
        &self,
        input: Matrix,
    ) -> (Vec<Matrix>, Vec<Matrix>, Vec<Matrix>) {
        // Step1: Linear projections
        let query = input.dot(&self.w_q) + self.b_q;
        let key = input.dot(&self.w_k) + self.b_k;
        let value = input.dot(&self.w_v) + self.b_v;

        let query_heads: Vec<Matrix> = self.get_heads(query);
        let key_heads: Vec<Matrix> = self.get_heads(key);
        let value_heads: Vec<Matrix> = self.get_heads(value);
        (query_heads, key_heads, value_heads)
    }

    fn get_heads(&self, input: Matrix) -> Vec<Matrix> {
        let head_dimension = self.dimension / self.nb_heads;
        (0..self.nb_heads)
            .map(|i| {
                input
                    .slice(s![.., i * head_dimension..(i + 1) * head_dimension])
                    .to_owned()
            })
            .collect()
    }

    pub fn scaled_dot_product(&self, query: Matrix, key: Matrix) -> Matrix {
        let head_dimension = self.dimension / self.nb_heads;
        query.dot(&key.t()) / (head_dimension as f32).sqrt()
    }

    pub fn output_projection(&self, heads: Vec<Matrix>) -> Matrix {
        let views: Vec<_> = heads.iter().map(|head| head.view()).collect();
        dbg!(&views);
        let concat_heads = concatenate(Axis(1), &views).expect("Heads have mismatching shapes");
        concat_heads.dot(&self.w_o) + &self.b_o
    }
}

pub fn softmax(input: Matrix) -> Matrix {
    let mut output = input.clone();
    for mut row in output.axis_iter_mut(Axis(0)) {
        let row_max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let denominator: f32 = row.iter().map(|v| (*v - row_max).exp()).sum();
        row.mapv_inplace(|v| (v - row_max).exp() / denominator);
    }
    output
}

pub struct LinearLayer {
    pub w: Matrix,
    pub b: Array1<f32>,
}

impl LinearLayer {
    pub fn run(&self, input: Matrix) -> Matrix {
        input.dot(&self.w) + &self.b
    }
}

pub enum ActivationFunction {
    GELU,
    ReLU,
    SiLU,
}

impl ActivationFunction {
    pub fn run(&self, input: f32) -> f32 {
        match self {
            ActivationFunction::GELU => input / (1.0 + (-1.702 * input).exp()),
            ActivationFunction::ReLU => todo!(),
            ActivationFunction::SiLU => input / (1.0 + (-input).exp()),
        }
    }
}

pub fn gelu(x: f32) -> f32 {
    x / (1.0 + (-1.702 * x).exp())
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[derive(Error, Debug)]
pub enum TransformerError {
    #[error("Dimension are not correct")]
    InvalidDimension(String),
}

#[cfg(test)]
mod tests {
    use crate::transformer_block::{
        AttentionLayer, LinearLayer, Matrix, NormalizationLayer, softmax,
    };
    use ndarray::{Array1, Array2, array};

    #[test]
    fn layer_normalization_simple() {
        let x = array![
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ];

        let layer_norm = NormalizationLayer {
            beta: array![0.0, 0.0],
            gamma: array![1.0, 1.0],
            epsilon: 1e-5,
        };

        let x_out = layer_norm.run(x).unwrap();
        let expected = array![
            [-1.46f32, -0.88, -0.29, 0.29, 0.88, 1.46],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ];
        assert_diff(&expected, &x_out, 1e-2);
    }

    #[test]
    fn linear_proj_and_heads_simple() {
        let x_in = array![[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0]];
        let mut attention = AttentionLayer::default();
        attention.dimension = 4;
        attention.nb_heads = 2;
        attention.w_q = Array2::eye(4);
        attention.w_k = Array2::zeros((4, 4));
        attention.w_v = Array2::ones((4, 4));
        attention.b_v = 1.0;

        let (queries, keys, values) = attention.linear_projections_and_heads(x_in);

        let expected_queries = vec![
            array![[0.1, 0.2], [0.0, 0.0]],
            array![[0.3, 0.4], [0.0, 0.0]],
        ];
        let expected_keys = vec![
            array![[0.0, 0.0], [0.0, 0.0]],
            array![[0.0, 0.0], [0.0, 0.0]],
        ];

        let expected_values = vec![
            array![[2.0, 2.0], [1.0, 1.0]],
            array![[2.0, 2.0], [1.0, 1.0]],
        ];

        assert_eq!(queries, expected_queries);
        assert_eq!(keys, expected_keys);
        assert_eq!(values, expected_values);
    }

    #[test]
    fn scaled_dot_product_simple() {
        let mut attention = AttentionLayer::default();
        attention.dimension = 6;
        attention.nb_heads = 2;
        let query = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let key = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        let out = attention.scaled_dot_product(query, key);
        let expected = array![[0.577, 0.0], [0.0, 0.577]];

        assert_diff(&out, &expected, 1e-2);
    }

    #[test]
    fn softmax_simple() {
        let x = Array2::ones((4, 4));
        let expected = Array2::ones((4, 4)) * 0.25;
        assert_diff(&softmax(x), &expected, 1e-2);
        let x = array![[0.577, 0.0], [0.0, 0.577]];
        let expected = array![[0.641, 0.359], [0.359, 0.641]];
        assert_diff(&softmax(x), &expected, 1e-2);
    }

    #[test]
    fn output_projection_simple() {
        let mut attention = AttentionLayer::default();
        attention.w_o = Array2::eye(4);
        attention.b_o = array![0.1, 0.2, 0.3, 0.4];

        let head1 = Array2::ones((2, 2));
        let head2 = Array2::eye(2);
        let expected = array![[1.1, 1.2, 1.3, 0.4], [1.1, 1.2, 0.3, 1.4]];

        let out = attention.output_projection(vec![head1, head2]);
        assert_diff(&expected, &out, 1e-3);
    }

    #[test]
    fn linear_layer_simple() {
        let x = Array2::ones((2, 2));
        let ll = LinearLayer {
            w: Array2::ones((2, 4)),
            b: Array1::ones(4),
        };

        let out = ll.run(x);
        let expected = Array2::ones((2, 4)) * 3.0;
        assert_diff(&out, &expected, 1e-2);
    }

    fn assert_diff(a: &Matrix, b: &Matrix, eps: f32) {
        assert_eq!(a.shape(), b.shape());
        let mut all_pass = true;
        for (((i, j), va), vb) in a.indexed_iter().zip(b.iter()) {
            let diff = (va - vb).abs();
            let pass = diff < eps;
            if !pass {
                println!(
                    "[{},{}]  a={:.6}  b={:.6}  diff={:.6}  FAIL",
                    i, j, va, vb, diff
                );
                all_pass = false;
            }
        }
        if !all_pass {
            panic!("Failed to pass")
        }
    }
}
