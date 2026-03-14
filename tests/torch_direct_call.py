import json
import sys
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._refs import Tensor


def parse_ndarray(obj) -> Tensor:
    """Parse a serialized Rust ndarray into a torch tensor."""
    return torch.tensor(obj["data"], dtype=torch.float32).reshape(obj["dim"])


def activation(name) -> Callable[[Tensor], Tensor]:
    """Matches the Rust sigmoid-approximation GELU: x / (1 + exp(-1.702x))"""
    if name == "GELU":
        return F.gelu
    elif name == "SiLU":
        return F.silu
    elif name == "ReLU":
        return F.relu
    else:
        raise ValueError(f"Unknown activation function: {name}")


def main():
    data = json.load(sys.stdin)

    x = parse_ndarray(data["input"])
    params = data["transformer_params"]

    attn = params["attention"]
    dim = attn["dimension"]
    nb_heads = attn["nb_heads"]
    ff_dim = params["linear_1"]["w"]["dim"][0]  # out dimension of linear_1
    activation_fn = activation(params["activation"])

    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=nb_heads,
        dim_feedforward=ff_dim,
        activation=activation_fn,
        batch_first=True,
        norm_first=True,
    )

    # Attention weights
    w_q = parse_ndarray(attn["w_q"])
    w_k = parse_ndarray(attn["w_k"])
    w_v = parse_ndarray(attn["w_v"])
    b_q = parse_ndarray(attn["b_q"])
    b_k = parse_ndarray(attn["b_k"])
    b_v = parse_ndarray(attn["b_v"])
    layer.self_attn.in_proj_weight = nn.Parameter(torch.cat([w_q, w_k, w_v], dim=0))
    layer.self_attn.in_proj_bias = nn.Parameter(torch.cat([b_q, b_k, b_v], dim=0))

    # Attention output projection
    layer.self_attn.out_proj.weight = nn.Parameter(parse_ndarray(attn["w_o"]))
    layer.self_attn.out_proj.bias = nn.Parameter(parse_ndarray(attn["b_o"]))

    # Feed-forward
    layer.linear1.weight = nn.Parameter(parse_ndarray(params["linear_1"]["w"]))
    layer.linear1.bias = nn.Parameter(parse_ndarray(params["linear_1"]["b"]))
    layer.linear2.weight = nn.Parameter(parse_ndarray(params["linear_2"]["w"]))
    layer.linear2.bias = nn.Parameter(parse_ndarray(params["linear_2"]["b"]))

    # Layer norms (PyTorch calls them weight/bias, Rust calls them gamma/beta)
    layer.norm1.weight = nn.Parameter(parse_ndarray(params["normalization_1"]["gamma"]))
    layer.norm1.bias = nn.Parameter(parse_ndarray(params["normalization_1"]["beta"]))
    layer.norm2.weight = nn.Parameter(parse_ndarray(params["normalization_2"]["gamma"]))
    layer.norm2.bias = nn.Parameter(parse_ndarray(params["normalization_2"]["beta"]))

    # Run
    layer.eval()
    with torch.no_grad():
        output = layer(x.unsqueeze(0)).squeeze(0)

    layer.eval()
    with torch.no_grad():
        x_in = x.unsqueeze(0)

        # Step by step (replicating what TransformerEncoderLayer does with norm_first=True)
        norm1_out = layer.norm1(x_in)
        attn_out, _ = layer.self_attn(norm1_out, norm1_out, norm1_out)
        residual_add_1 = x_in + attn_out

        norm2_out = layer.norm2(residual_add_1)
        lin1_out = layer.linear1(norm2_out)
        activation_out = activation_fn(lin1_out)
        lin2_out = layer.linear2(activation_out)
        final_out = residual_add_1 + lin2_out

        print(
            json.dumps(
                {
                    "norm1_out": norm1_out.squeeze(0).tolist(),
                    "attn_out": attn_out.squeeze(0).tolist(),
                    "residual1": residual_add_1.squeeze(0).tolist(),
                    "norm2_out": norm2_out.squeeze(0).tolist(),
                    "lin1_out": lin1_out.squeeze(0).tolist(),
                    "act_out": activation_out.squeeze(0).tolist(),
                    "lin2_out": lin2_out.squeeze(0).tolist(),
                    "final_out": final_out.squeeze(0).tolist(),
                    "output": output.squeeze(0).tolist(),
                }
            )
        )


if __name__ == "__main__":
    main()
