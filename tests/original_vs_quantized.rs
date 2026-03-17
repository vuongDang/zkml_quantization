use ark_bn254::Fr;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand::distr::Uniform;
use simple_zkml::quantization_schemes::fixed_field::FixedPointQuantization;
use simple_zkml::quantized_transformer::QuantizedTransformerBlock;
use simple_zkml::transformer_block::TransformerBlock;

#[test]
fn original_vs_quantized_model_simple() {
    let d_model = 4;
    let original_model = TransformerBlock::random(d_model, 2, 6);

    let qs = FixedPointQuantization {
        frac_bits: 8,
        total_bits: 16,
    };
    let quantized_model =
        QuantizedTransformerBlock::<Fr, FixedPointQuantization>::new(original_model.clone(), qs);

    let dist = Uniform::new(-3.0, 3.0).unwrap();
    let input = Array2::random((2, d_model), dist);

    let original_out = original_model.run(input.clone()).unwrap();
    let quantized_out = quantized_model.run(input).unwrap();

    dbg!(original_out);
    dbg!(quantized_out);
}
