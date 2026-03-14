use std::{
    collections::HashMap,
    io::Write,
    process::{Command, Stdio},
};

use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand::distr::Uniform;
use serde_json::json;
use simple_zkml::transformer_block::*;

#[test]
fn test_against_pytorch() {
    let d_model = 4;
    for _ in (0..5).into_iter() {
        let block = TransformerBlock::random(d_model, 2, 6);
        let dist = Uniform::new(-3.0, 3.0).unwrap();
        let input = Array2::random((2, d_model), dist);

        let json_input = json!(input);
        let json_block = json!(block);

        let params = json!({
            "input": json_input,
            "transformer_params": json_block
        });

        let python_output_1 = call_pytorch_impl("tests/torch_steps_impl.py", params.clone());

        let python_output_2 = call_pytorch_impl("tests/torch_direct_call.py", params);
        // dbg!(&python_output_2);

        let rust_output: Array2<f32> = block.run(input).unwrap();
        // .rows()
        // .into_iter()
        // .map(|row| row.to_vec())
        // .collect();

        assert_diff(&rust_output, &python_output_1["final_out"], 1e-2);
        assert_diff(&rust_output, &python_output_2["final_out"], 1e-2);
    }
}

fn call_pytorch_impl(script: &str, params: serde_json::Value) -> HashMap<String, Array2<f32>> {
    let mut python_transformer = Command::new("python")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to start python");

    python_transformer
        .stdin
        .take()
        .unwrap()
        .write_all(params.to_string().as_bytes())
        .unwrap();

    let output = python_transformer.wait_with_output().unwrap();
    assert!(&output.status.success());

    let map: HashMap<String, Vec<Vec<f32>>> = serde_json::from_slice(&output.stdout).unwrap();
    map.into_iter()
        .map(|(k, v)| {
            let rows = v.len();
            let cols = v.first().unwrap().len();
            let flat = v.into_iter().flatten().collect();
            (k, Array2::from_shape_vec((rows, cols), flat).unwrap())
        })
        .collect()
}

fn assert_diff(a: &Array2<f32>, b: &Array2<f32>, eps: f32) {
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
