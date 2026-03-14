import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._refs import Tensor


def parse_ndarray(obj) -> Tensor:
    """Parse a serialized Rust ndarray into a torch tensor."""
    return torch.tensor(obj["data"], dtype=torch.float32).reshape(obj["dim"])


def normalization_layer(params, input, dim) -> Tensor:
    gamma = parse_ndarray(params["gamma"])
    beta = parse_ndarray(params["beta"])
    epsilon = params["epsilon"]
    return F.layer_norm(input, [dim], gamma, beta, eps=epsilon)


def multihead_attention(params, input) -> Tensor:
    dim = params["dimension"]
    nb_heads = params["nb_heads"]
    w_q = parse_ndarray(params["w_q"])
    w_k = parse_ndarray(params["w_k"])
    w_v = parse_ndarray(params["w_v"])
    b_q = parse_ndarray(params["b_q"])
    b_k = parse_ndarray(params["b_k"])
    b_v = parse_ndarray(params["b_v"])

    w_o = parse_ndarray(params["w_o"])
    b_o = parse_ndarray(params["b_o"])

    attn = nn.MultiheadAttention(dim, nb_heads, batch_first=True)
    attn.in_proj_weight = nn.Parameter(torch.concat([w_q, w_k, w_v]))
    attn.in_proj_bias = nn.Parameter(torch.concat([b_q, b_k, b_v]))
    attn.out_proj.weight = nn.Parameter(w_o)
    attn.out_proj.bias = nn.Parameter(b_o)
    return attn(input, input, input)


def linear_layer(params, input) -> Tensor:
    w = parse_ndarray(params["w"])
    b = parse_ndarray(params["b"])
    return input @ w.T + b


def activation(name, x) -> Tensor:
    """Matches the Rust sigmoid-approximation GELU: x / (1 + exp(-1.702x))"""
    if name == "GELU":
        return F.gelu(x)
    elif name == "SiLU":
        return F.silu(x)
    elif name == "ReLU":
        return F.relu(x)
    else:
        raise ValueError(f"Unknown activation function: {name}")


def main():
    data = json.load(sys.stdin)
    # print(json.dumps(data, indent=2), file=sys.stderr)

    x = parse_ndarray(data["input"])
    params = data["transformer_params"]
    dim = params["attention"]["dimension"]

    norm1_out = normalization_layer(params["normalization_1"], x, dim)
    attn_out, _ = multihead_attention(params["attention"], norm1_out)
    residual_add_1 = x + attn_out

    norm2_out = normalization_layer(params["normalization_2"], residual_add_1, dim)
    lin1_out = linear_layer(params["linear_1"], norm2_out)
    activation_out = activation(params["activation"], lin1_out)
    lin2_out = linear_layer(params["linear_2"], activation_out)
    final_out = residual_add_1 + lin2_out

    result = {
        "norm1_out": norm1_out.tolist(),
        "attn_out": attn_out.tolist(),
        "residual_add_1": residual_add_1.tolist(),
        "norm2_out": norm2_out.tolist(),
        "lin1_out": lin1_out.tolist(),
        "activation_out": activation_out.tolist(),
        "lin2_out": lin2_out.tolist(),
        "final_out": final_out.tolist(),
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
