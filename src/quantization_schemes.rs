use ark_ff::PrimeField;
use ndarray::Array2;

pub trait QuantizationScheme<F: PrimeField>: Clone {
    fn quantize(&self, x: f32) -> F;
    fn dequantize(&self, x: F) -> f32;
    // Maximum positive integer this scheme can represent
    fn max_int(&self) -> f32;

    // Compute the scaling of a matrix
    fn compute_scale(&self, data: &Array2<f32>) -> f32 {
        let max_value = data
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, |acc, v| f32::max(acc, v));
        max_value / self.max_int()
    }
}
