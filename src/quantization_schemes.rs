pub mod fixed_field;

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

pub fn field_to_i64<F: PrimeField>(v: F) -> i64 {
    // if v > (p-1)/2 it represents a negative number
    let bits = v.into_bigint();
    let limbs = bits.as_ref();

    // check if value fit in 64 bits
    let fits_in_u64 = limbs[1..].iter().all(|&x| x == 0);
    assert!(
        fits_in_u64,
        "field element does not fit in i64 — quantization overflow? value has {} limbs",
        limbs.len()
    );

    if bits > F::MODULUS_MINUS_ONE_DIV_TWO {
        let neg = (-v).into_bigint();
        -(neg.as_ref()[0] as i64)
    } else {
        bits.as_ref()[0] as i64
    }
}

#[cfg(test)]
mod tests {
    use crate::quantization_schemes::field_to_i64;

    #[test]
    fn test_field_to_i64_fits_in_limb() {
        use ark_bn254::Fr;

        // these should all work — small values fit in one limb of bigint
        for v in [0i64, 1, -1, 127, -127, 32767, -32767] {
            let f: Fr = if v >= 0 {
                Fr::from(v as u64)
            } else {
                -Fr::from((-v) as u64)
            };
            assert_eq!(field_to_i64(f), v);
        }
    }

    #[test]
    #[should_panic]
    fn test_field_to_i64_panics_on_large_value() {
        use ark_bn254::Fr;

        // construct a value that spans multiple limbs
        // take a large BN254 field element that cannot fit in i64
        let large = Fr::from(u64::MAX) * Fr::from(u64::MAX);
        field_to_i64(large); // should panic
    }
}
