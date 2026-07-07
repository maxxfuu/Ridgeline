import torch

DTYPE_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}

def linear_flops(batch: int, in_features: int, out_features: int) -> int:
    # M * N * (2*K)
    return batch * out_features * (2 * in_features)

def linear_bytes(batch: int, in_features: int, out_features: int, dtype_bytes: int) -> int:
    # All of the cells converted to dtype_bytes 
    return dtype_bytes * ((batch * in_features) + (in_features * out_features) + (batch * out_features))

def lm_head_flops(batch: int, query_len: int, d_model: int, vocab_size: int) -> int:
      # the final Linear(d_model -> vocab_size) that produces logits; runs once per forward
      return linear_flops(batch * query_len, d_model, vocab_size)

def lm_head_bytes(batch: int, query_len: int, d_model: int, vocab_size: int, dtype_bytes: int) -> int:
      return linear_bytes(batch * query_len, d_model, vocab_size, dtype_bytes)

def attention_flops(batch: int, query_len: int, ctx_len: int, d_model: int, n_heads: int) -> int:
    # 2 matmuls (Q@K^T, attn_weights@V), each 2 FLOPs/element (mult+add),
    # summed across all heads and the batch
    head_dim = d_model // n_heads 
    return batch * (n_heads * 2 * 2 * (query_len * ctx_len) * head_dim)

def attention_bytes(batch: int, query_len: int, ctx_len: int, d_model: int, n_heads: int, dtype_bytes: int) -> int:
    io_bytes = dtype_bytes * batch * d_model * (2 * ctx_len + 2 * query_len)   # Q, K, V, out
    scores_bytes = dtype_bytes * batch * n_heads * query_len * ctx_len * 2     # write + read back
    return io_bytes + scores_bytes

def predict_layer(cfg: dict, batch: int, query_len: int, ctx_len: int, dtype_bytes: int) -> dict:
    d_model = cfg["emb_dim"]
    n_heads = cfg["n_heads"]
    rows = batch * query_len  # tokens actually fed through each linear this call

    # W_Q, W_K, W_V, out_proj: 4 identical (d_model -> d_model) linears
    qkvo_flops = 4 * linear_flops(rows, d_model, d_model)
    qkvo_bytes = 4 * linear_bytes(rows, d_model, d_model, dtype_bytes)

    # attention math itself: Q@K^T and attn_weights@V
    attn_flops = attention_flops(batch, query_len, ctx_len, d_model, n_heads)
    attn_bytes = attention_bytes(batch, query_len, ctx_len, d_model, n_heads, dtype_bytes)

    # FFN: up (d_model -> 4*d_model) and down (4*d_model -> d_model), same cost either way
    ffn_flops = 2 * linear_flops(rows, d_model, 4 * d_model)
    ffn_bytes = 2 * linear_bytes(rows, d_model, 4 * d_model, dtype_bytes)

    return {
        "flops": qkvo_flops + attn_flops + ffn_flops,
        "bytes": qkvo_bytes + attn_bytes + ffn_bytes,
    }

def predict_model(cfg: dict, batch: int, query_len: int, ctx_len: int, dtype_bytes: int) -> dict:
      layer = predict_layer(cfg, batch, query_len, ctx_len, dtype_bytes)
      n_layers = cfg["n_layers"]
      head_flops = lm_head_flops(batch, query_len, cfg["emb_dim"], cfg["vocab_size"])
      head_bytes = lm_head_bytes(batch, query_len, cfg["emb_dim"], cfg["vocab_size"], dtype_bytes)
      return {
          "flops": layer["flops"] * n_layers + head_flops,
          "bytes": layer["bytes"] * n_layers + head_bytes,
      }