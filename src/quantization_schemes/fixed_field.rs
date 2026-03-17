use ark_ff::PrimeField;

use crate::quantization_schemes::{QuantizationScheme, field_to_i64};

/// Fixed-point: x = q * 2^(-f)
/// f fracional bits
/// x float
/// q quantized integer
///
/// Pros: a simple right shift operation
#[derive(Debug, Clone)]
pub struct FixedPointQuantization {
    // Scaling and the number of bits to represent the represent
    // the fractional part
    // e.g 8 means scale = 2^(-8)
    pub frac_bits: u32,

    // The total number of bits used for our quantized integers
    pub total_bits: u32,
}

impl<F: PrimeField> QuantizationScheme<F> for FixedPointQuantization {
    fn quantize(&self, x: f32) -> F {
        let shifted = (x * ((1u32 << self.frac_bits) as f32)).round() as i64;

        // F::from only takes from unsigned integer
        if shifted >= 0 {
            F::from(shifted as u64)
        } else {
            F::zero() - F::from((-shifted) as u64)
        }
    }

    fn dequantize(&self, q: F) -> f32 {
        let int_val = field_to_i64(q);
        (int_val as f32) / ((1u32 << self.frac_bits) as f32)
    }

    // The biggest int we want the scheme to have
    fn max_int(&self) -> f32 {
        ((1u64 << (self.total_bits - 1)) - 1) as f32
    }
}
