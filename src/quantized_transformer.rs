use crate::{
    quantization_schemes::QuantizationScheme,
    transformer_block::{
        ActivationFunction, AttentionLayer, LinearLayer, NormalizationLayer, TransformerBlock,
        TransformerError, softmax,
    },
};
use ark_ff::PrimeField;
use ndarray::{Array1, Array2, Axis, s};

pub struct QuantizedTransformerBlock<F, QS>
where
    F: PrimeField,
    QS: QuantizationScheme<F>,
{
    scheme: QS,
    normalization_1: QuantizedNormalizationLayer<F>,
    attention: QuantizedAttentionLayer<F, QS>,
    normalization_2: QuantizedNormalizationLayer<F>,
    linear_1: QuantizedLinearLayer<F, QS>,
    activation: ActivationFunction,
    linear_2: QuantizedLinearLayer<F, QS>,
}

impl<F: PrimeField, QS: QuantizationScheme<F>> QuantizedTransformerBlock<F, QS> {
    pub fn new(block: TransformerBlock, qs: QS) -> Self {
        QuantizedTransformerBlock {
            scheme: qs.clone(),
            normalization_1: QuantizedNormalizationLayer::new(block.normalization_1, &qs),
            attention: QuantizedAttentionLayer::new(block.attention, &qs),
            normalization_2: QuantizedNormalizationLayer::new(block.normalization_2, &qs),
            linear_1: QuantizedLinearLayer::new(block.linear_1, &qs),
            activation: block.activation,
            linear_2: QuantizedLinearLayer::new(block.linear_2, &qs),
        }
    }

    pub fn run(&self, input: Array2<f32>) -> Result<Array2<f32>, TransformerError> {
        // Quantize the input
        let quantized_input = input.mapv(|x| self.scheme.quantize(x));

        // Normalization 1
        let norm1_out = self.normalization_1.run(quantized_input.clone());

        // Attention
        let attention_out = self.attention.run(&norm1_out);

        // Residual Addition
        let residual1_out = quantized_input + attention_out;
        // println!("residual1: {:?}", &residual1_out);

        // Feed-forward Block
        // Normalization
        let norm2_out = self.normalization_2.run(residual1_out.clone());
        // println!("norm2: {:?}", &norm2_out);

        // LinearLayer
        let mut lin1_out = self.linear_1.run(norm2_out);
        // println!("lin1: {:?}", &lin1_out);

        // Activation Layer
        lin1_out.mapv_inplace(|elem| self.activation.quantized_run(elem, &self.scheme));
        // println!("activation: {:?}", &lin1_out);

        // LinearLayer
        let lin2_out = self.linear_2.run(lin1_out);
        // println!("lin2: {:?}", &lin2_out);

        // Residual Addition
        let final_out = residual1_out + lin2_out;
        // println!("final: {:?}", &final_out);

        let dequantized_out = final_out.mapv(|x| self.scheme.dequantize(x));

        Ok(dequantized_out)
    }

    pub fn random(scheme: &QS, d_model: usize, nb_heads: usize, d_ffn: usize) -> Self {
        let original = TransformerBlock::random(d_model, nb_heads, d_ffn);
        Self::new(original, scheme.clone())
    }
}

pub struct QuantizedNormalizationLayer<F: PrimeField> {
    pub beta: Array1<F>,
    pub gamma: Array1<F>,
    pub epsilon: F,
}

impl<F: PrimeField> QuantizedNormalizationLayer<F> {
    pub fn new(layer: NormalizationLayer, qs: &impl QuantizationScheme<F>) -> Self {
        let beta = layer.beta.mapv(|x| qs.quantize(x));
        let gamma = layer.gamma.mapv(|x| qs.quantize(x));
        let epsilon = qs.quantize(layer.epsilon);
        QuantizedNormalizationLayer {
            beta,
            gamma,
            epsilon,
        }
    }

    // Layer Formula
    // output = gamma * (x - mean) / sqrt(variance + epsilon) + beta
    pub fn run(&self, input: Array2<F>) -> Array2<F> {
        let dim = input.ncols();
        let inv_dim = F::from(dim as u64)
            .inverse()
            .expect("dimension must be non-zero");
        let mut output = Array2::zeros(input.dim());
        for (i, row) in input.axis_iter(Axis(0)).enumerate() {
            // Mean
            let sum: F = row.iter().copied().fold(F::zero(), |acc, elem| acc + elem);
            let mean = sum * inv_dim;

            // Centered values
            let centered = row.iter().map(|&elem| elem - mean);

            // Variance
            let variance = centered
                .clone()
                .fold(F::zero(), |acc, elem| acc + elem * elem)
                * inv_dim;

            dbg!(&variance);

            // Inverse of standard deviation
            let inv_std = layer_norm_sqrt(variance, self.epsilon)
                .inverse()
                .expect("must be non-zero");

            for (j, val) in centered.enumerate() {
                output[[i, j]] = self.gamma[j] * val * inv_std + self.beta[j];
            }
        }
        output
    }
}

fn layer_norm_sqrt<F: PrimeField>(variance: F, epsilon: F) -> F {
    let mut var_eps = variance + epsilon;
    for attempt in 0..100 {
        match var_eps.sqrt() {
            Some(v) => {
                return v.inverse().expect("must be non-zero");
            }
            None => {
                if attempt > 10 {
                    panic!(
                        "could not find sqrt after {} nudges, check epsilon scale",
                        attempt
                    );
                } else {
                    var_eps = var_eps + epsilon; // nudge by epsilon until we hit a qr
                }
            }
        }
    }
    unreachable!()
}

pub struct QuantizedAttentionLayer<F, QS>
where
    F: PrimeField,
    QS: QuantizationScheme<F>,
{
    pub dimension: usize,
    pub nb_heads: usize,
    pub scheme: QS,

    // Query, Key and Value weights and biases
    pub w_q: Array2<F>,
    pub b_q: Array1<F>,
    pub w_k: Array2<F>,
    pub b_k: Array1<F>,
    pub w_v: Array2<F>,
    pub b_v: Array1<F>,
    pub w_o: Array2<F>,
    pub b_o: Array1<F>,
    pub inv_rescale_qk: F,
    pub inv_rescale_qv: F,
    pub scale_q: f32,
    pub scale_k: f32,
    pub scale_v: f32,
    pub scale_o: f32,
}

impl<F: PrimeField, QS: QuantizationScheme<F>> QuantizedAttentionLayer<F, QS> {
    pub fn new(layer: AttentionLayer, scheme: &QS) -> Self {
        let w_q = layer.w_q.mapv(|x| scheme.quantize(x));
        let b_q = layer.b_q.mapv(|x| scheme.quantize(x));
        let w_k = layer.w_k.mapv(|x| scheme.quantize(x));
        let b_k = layer.b_k.mapv(|x| scheme.quantize(x));
        let w_v = layer.w_v.mapv(|x| scheme.quantize(x));
        let b_v = layer.b_v.mapv(|x| scheme.quantize(x));
        let w_o = layer.w_o.mapv(|x| scheme.quantize(x));
        let b_o = layer.b_o.mapv(|x| scheme.quantize(x));

        let scale_q = scheme.compute_scale(&layer.w_q);
        let scale_k = scheme.compute_scale(&layer.w_k);
        let scale_v = scheme.compute_scale(&layer.w_v);
        let scale_o = scheme.compute_scale(&layer.w_o);

        let inv_rescale_qk = inv_rescale(scale_q, scale_k, scale_q);
        let inv_rescale_qv = inv_rescale(scale_q, scale_v, scale_v);
        Self {
            dimension: layer.dimension,
            nb_heads: layer.nb_heads,
            scheme: scheme.clone(),
            w_q,
            b_q,
            w_k,
            b_k,
            w_v,
            b_v,
            w_o,
            b_o,
            scale_q,
            scale_k,
            scale_v,
            scale_o,
            inv_rescale_qk,
            inv_rescale_qv,
        }
    }

    pub fn run(&self, input: &Array2<F>) -> Array2<F> {
        let input_f32 = input.mapv(|x| self.scheme.dequantize(x));
        let scale_input = self.scheme.compute_scale(&input_f32);

        let inv_rescale_q: F = inv_rescale(scale_input, self.scale_q, self.scale_q);
        let inv_rescale_k: F = inv_rescale(scale_input, self.scale_k, self.scale_k);
        let inv_rescale_v: F = inv_rescale(scale_input, self.scale_v, self.scale_v);
        let inv_rescale_o: F = inv_rescale(scale_input, self.scale_o, self.scale_o);

        // Linear projections and split into heads
        let (queries, keys, values) = self.linear_projections_and_heads(
            input,
            &inv_rescale_q,
            &inv_rescale_k,
            &inv_rescale_v,
        );

        // Parallel processing of heads
        let mut heads = vec![];
        for (q, (k, v)) in queries
            .into_iter()
            .zip(keys.into_iter().zip(values.into_iter()))
        {
            // Scaled dot products
            let score = self.scaled_dot_product(q, k);

            // softmax
            let softmax_score = quantized_softmax(score, &self.scheme);

            // Linear layer with values
            let out = softmax_score.dot(&v);
            let out_rescaled = out.mapv(|x| x * self.inv_rescale_qv);
            heads.push(out_rescaled);
        }
        // Output projection
        self.output_projection(heads, &inv_rescale_o)
    }

    pub fn linear_projections_and_heads(
        &self,
        input: &Array2<F>,
        inv_rescale_q: &F,
        inv_rescale_k: &F,
        inv_rescale_v: &F,
    ) -> (Vec<Array2<F>>, Vec<Array2<F>>, Vec<Array2<F>>) {
        // Step1: Linear projections
        let query = quantized_linear(&input, &self.w_q, &self.b_q, inv_rescale_q);
        let key = quantized_linear(&input, &self.w_k, &self.b_k, inv_rescale_k);
        let value = quantized_linear(&input, &self.w_v, &self.b_v, inv_rescale_v);

        let query_heads = self.get_heads(query);
        let key_heads = self.get_heads(key);
        let value_heads = self.get_heads(value);
        (query_heads, key_heads, value_heads)
    }

    // This does not change from the original version
    fn get_heads(&self, input: Array2<F>) -> Vec<Array2<F>> {
        let head_dimension = self.dimension / self.nb_heads;
        (0..self.nb_heads)
            .map(|i| {
                input
                    .slice(s![.., i * head_dimension..(i + 1) * head_dimension])
                    .to_owned()
            })
            .collect()
    }

    pub fn scaled_dot_product(&self, query: Array2<F>, key: Array2<F>) -> Array2<F> {
        let head_dimension_sqrt = (self.dimension / self.nb_heads).isqrt();
        let inv = F::from(head_dimension_sqrt as u64)
            .inverse()
            .expect("head_dimension should be invertible");

        let scores = query.dot(&key.t());
        scores.mapv_into(|elem| elem * self.inv_rescale_qk * inv)
    }

    pub fn output_projection(&self, heads: Vec<Array2<F>>, inv_rescale_o: &F) -> Array2<F> {
        let views: Vec<_> = heads.iter().map(|head| head.view()).collect();
        let concat_heads =
            ndarray::concatenate(Axis(1), &views).expect("Heads have mismatching shapes");
        quantized_linear(&concat_heads, &self.w_o, &self.b_o, inv_rescale_o)
    }
}

pub struct QuantizedLinearLayer<F: PrimeField, QS: QuantizationScheme<F>> {
    pub w: Array2<F>,
    pub b: Array1<F>,
    pub scale_w: f32,
    pub scheme: QS,
}

impl<F: PrimeField, QS: QuantizationScheme<F>> QuantizedLinearLayer<F, QS> {
    pub fn new(layer: LinearLayer, qs: &QS) -> Self {
        let scale_w = qs.compute_scale(&layer.w);
        let w = layer.w.mapv(|x| qs.quantize(x / scale_w));
        let b = layer.b.mapv(|x| qs.quantize(x));
        Self {
            w,
            b,
            scale_w,
            scheme: qs.clone(),
        }
    }

    pub fn run(&self, input: Array2<F>) -> Array2<F> {
        let input_f32 = input.mapv(|x| self.scheme.dequantize(x));
        let scale_input = self.scheme.compute_scale(&input_f32);
        let inv_rescale: F = inv_rescale(scale_input, self.scale_w, self.scale_w);

        quantized_linear(&input, &self.w, &self.b, &inv_rescale)
    }
}

impl ActivationFunction {
    pub fn quantized_run<F: PrimeField>(&self, input: F, qs: &impl QuantizationScheme<F>) -> F {
        let dequantized_input = qs.dequantize(input);
        let activation_out = self.run(dequantized_input);
        qs.quantize(activation_out)
    }
}

// Either:
// - table lookup
// - taylor series
// - dequantize -> softmax -> quantize
// We don't really care here because no imprecision error comes
// from this step
pub fn quantized_softmax<F: PrimeField, QS: QuantizationScheme<F>>(
    input: Array2<F>,
    quant_scheme: &QS,
) -> Array2<F> {
    let dequantized_input = input.mapv(|x| quant_scheme.dequantize(x));
    let softmax_out = softmax(dequantized_input);
    softmax_out.mapv(|x| quant_scheme.quantize(x))
}

fn quantized_linear<F: PrimeField>(
    input: &Array2<F>,
    product: &Array2<F>,
    bias: &Array1<F>,
    inv_rescale: &F,
) -> Array2<F> {
    let product = input.dot(&product.t());
    let rescale = product.mapv(|x| x * inv_rescale);
    rescale + bias
}

// When doing multiplication a.dot(b) in ff, result should be scale
// down to scale_out.
fn inv_rescale<F: PrimeField>(scale_a: f32, scale_b: f32, scale_out: f32) -> F {
    let ratio = scale_out / (scale_a * scale_b);
    // Get the ratio as a fraction
    let shift = 32u32;
    let numerator = (ratio * (1u64 << shift) as f32).round() as u64;
    F::from(numerator) * F::from(1u64 << shift).inverse().unwrap()
}
