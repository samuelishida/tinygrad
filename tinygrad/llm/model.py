from __future__ import annotations
import enum, functools, itertools, pathlib
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes
from tinygrad.llm.kernels.amd import Linear, gated_delta_prefill, flash_attention, amd_custom_kernels_supported, kv_q8_quantize, kv_q8_quantize_batched, kv_q8_dequant, quant_raw_info, expert_linear, Q8_GROUP_SIZE, MOE_FUSED_DECODE
from tinygrad.llm.gguf import gguf_load
from tinygrad.uop.ops import resolve

MTP_TMAX = 32  # max MTP verify batch; must be a multiple of the flash BLOCK_M

class ExpertGating(enum.IntEnum):
  SOFTMAX = 1
  SIGMOID = 2
  SOFTMAX_WEIGHT = 3  # softmax over the top-k selected logits
  SQRT_SOFTPLUS = 4

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device)

class ExpertWeights:
  """Like Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
    self.out_features, self.in_features = out_features, in_features
    self._packed = None  # probe once: (packed_uop, ggml_type) or False (unsupported)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    # fused expert GEMM (AMD RDNA3): gather-on-load from the packed ggml buffer —
    # no fp16 materialization of the selected experts (the generic path below
    # writes+reads ~8x more DRAM per decode token).
    if self._packed is None:
      if amd_custom_kernels_supported(self.weight.device):
        info = quant_raw_info(self.weight)
        self._packed = (info[2], info[1]) if info is not None else False
      else:
        self._packed = False
    if self._packed is not False and MOE_FUSED_DECODE and isinstance(sel.numel(), int):  # decode: static token count
      # pair (b,t,i) with the shared activation row (gate/up) or its own row (down)
      x2d = x.expand(*sel.shape, x.shape[-1]).reshape(-1, self.in_features)
      packed, ggml_type = self._packed
      out = expert_linear(sel.reshape(-1), packed, ggml_type, x2d, self.out_features)
      return out.reshape(*sel.shape, self.out_features)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int
  kda: bool = False

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  v_head_dim: int
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  expert_gating_func: ExpertGating = ExpertGating.SOFTMAX
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  ssm_layers: tuple[bool, ...] = ()
  attn_output_gate: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False

def _sub_stat(x:Tensor) -> Tensor:
  """Lazy (n_nan, n_inf, absmax) stat tensor — the probe realizes it with the logits
  in one realize call so the var binding flows (never realize it inside a jit fxn)."""
  return Tensor.stack(
    x.isnan().cast(dtypes.int32).sum().cast(dtypes.float32),
    x.isinf().cast(dtypes.int32).sum().cast(dtypes.float32),
    x.float().abs().max().cast(dtypes.float32))

# set to a list by Transformer._forward_hidden while _per_layer_debug is on:
# FFNBlock.__call__ then appends sub-part stats (post-attention, post-FFN) and
# bypasses the @function dispatch (identical math, no precompile boundary).
dbg_stats: list[Tensor] | None = None
dbg_sub: bool = False  # also collect post-attn/post-FFN sub stats (bigger capture graph)

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config
    self.use_flash = True

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=False)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      logits = self.ffn_gate_inp(x)
      bias = self.exp_probs_b["bias"] if hasattr(self, 'exp_probs_b') else None
      gating, normalize_topk = self.config.expert_gating_func, self.config.norm_topk_prob
      # fast path: without selection bias, normalized SOFTMAX is equivalent to SOFTMAX_WEIGHT
      if gating == ExpertGating.SOFTMAX and bias is None and normalize_topk:
        gating, normalize_topk = ExpertGating.SOFTMAX_WEIGHT, False
      if   gating == ExpertGating.SOFTMAX_WEIGHT: scores = logits
      elif gating == ExpertGating.SOFTMAX:        scores = logits.softmax(-1)
      elif gating == ExpertGating.SIGMOID:        scores = logits.sigmoid()
      elif gating == ExpertGating.SQRT_SOFTPLUS:  scores = logits.softplus().sqrt()

      _, sel = pairwise_topk(scores if bias is None else scores + bias, self.config.num_experts_per_tok)
      probs = scores.gather(-1, sel)
      # SOFTMAX_WEIGHT applies softmax after top-k selection
      if gating == ExpertGating.SOFTMAX_WEIGHT: probs = probs.softmax(-1)
      if normalize_topk: probs = probs / probs.sum(axis=-1, keepdim=True)
      probs = probs * self.config.routed_scaling_factor
      x_down = self.ffn_down_exps(sel, (self.ffn_gate_exps(sel, h).silu() * self.ffn_up_exps(sel, h)).contiguous())  # (B, T, k, D)
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(self.ffn_gate_shexp(x).silu().contiguous() * self.ffn_up_shexp(x))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    # TODO: remove the need for this contiguous
    return self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp):
    self._init_state(x)
    if dbg_stats is not None and dbg_sub:
      # debug detach of the @function dispatch: same math, plain tensor calls, with
      # sub-part stats so the diag probe can attribute a first-NaN to the attention
      # (flash-decode / GDN scan) or the FFN (MoE) of the block.
      h =     x + self._attention(self.attn_norm(x), start_pos)
      dbg_stats.append(_sub_stat(h))
      out =   h + self._feed_forward(self.ffn_norm(h))
      dbg_stats.append(_sub_stat(out))
      return out.contiguous()
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp):
      h =     x + self._attention(self.attn_norm(x), start_pos)
      return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    return _run(x, start_pos)

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = Linear(config.head_dim * config.n_heads, config.dim, bias=False)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    if self.config.attn_output_gate:
      qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
    k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    # Q8 KV write: quantize each 32-dim group to int8 + fp32 scale (4 bytes per
    # uint32 word); the flash kernels dequantize on read (int8(byte) * scale).
    # Batched (shape-generic) so it stays valid under symbolic draft counts. Gated on
    # _q8_kv: unsupported head sizes (tiny unit-test models) keep the plain fp16 cache.
    if self._q8_kv:
      kv_q8, kv_sc = kv_q8_quantize_batched(Tensor.stack(k, v))
      q8_store = self.cache_kv[:, :, :, start_pos:start_pos+T, :, :].uop.store(kv_q8.uop)
      sc_store = self.cache_kv_scale[:, :, :, start_pos:start_pos+T, :].uop.store(kv_sc.uop)
      assigned_kv = Tensor(self.cache_kv.uop.after(q8_store))
      assigned_kv_scale = Tensor(self.cache_kv_scale.uop.after(sc_store))
      # on RDNA3, hybrid models use custom flash attention kernels on the KV cache
      if amd_custom_kernels_supported(x.device) and self.config.ssm is not None and self.use_flash:
        attn = flash_attention(q, assigned_kv, assigned_kv_scale, start_pos+T)
        attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
        return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))
      k = kv_q8_dequant(assigned_kv[0, :, :, 0:start_pos+T], assigned_kv_scale[0, :, :, 0:start_pos+T])
      v = kv_q8_dequant(assigned_kv[1, :, :, 0:start_pos+T], assigned_kv_scale[1, :, :, 0:start_pos+T])
    else:
      store = self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).cast(dtypes.half).uop)
      assigned_kv = Tensor(self.cache_kv.uop.after(store))
      k, v = assigned_kv[0, :, :, 0:start_pos+T, :], assigned_kv[1, :, :, 0:start_pos+T, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # Q8 layout: packed uint32 words (2,B,KvH,N,8,8) + fp32 group scales
      # (2,B,KvH,N,8) — the flash kernels dequantize on read as int8(byte) * scale.
      if amd_custom_kernels_supported(x.device) and self.config.head_dim % Q8_GROUP_SIZE == 0:
        self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim//32, 8,
                                     dtype=dtypes.uint32, device=x.device)
        self.cache_kv_scale = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim//32,
                                           dtype=dtypes.float32, device=x.device)
        self._q8_kv = True
      else:
        # unsupported head size (e.g. tiny unit-test models): plain fp16 KV cache + standard SDPA
        self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim,
                                     dtype=dtypes.half, device=x.device)
        self._q8_kv = False
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    if not self.config.ssm or not self.config.ssm.kda: q_rope = apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T])
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(q_rope, dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2)
    if not self.config.ssm or not self.config.ssm.kda: k_rope = apply_rope(k_rope, self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.uop)))[:, :, 0:start_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class MTPBlock:
  """Multi-token-prediction (DeepSeek-V3 / Qwen3-Next) draft head.

  Fuses the embedding of the current token with the previous-position hidden state
  through `enorm`/`hnorm` + `eh_proj`, runs the result through a dedicated
  transformer block, and produces next-token logits via `shared_head_norm` + the
  shared `output` head. `load_from_gguf` consumes the `blk.<idx>.*` and
  `blk.<idx>.nextn.*` GGUF weights.
  """
  def __init__(self, config:TransformerConfig, block_cls:type[FFNBlock]):
    self.enorm = nn.RMSNorm(config.dim, config.norm_eps)
    self.hnorm = nn.RMSNorm(config.dim, config.norm_eps)
    self.eh_proj = Linear(2*config.dim, config.dim, bias=False)
    self.shared_head_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.block = block_cls(config)

  def fuse(self, emb:Tensor, hidden:Tensor) -> Tensor:
    return self.eh_proj(self.enorm(emb).cat(self.hnorm(hidden), dim=-1))

  def forward(self, fused:Tensor, start_pos:int|UOp) -> Tensor:
    return self.block(fused, start_pos)

  def load_from_gguf(self, idx:int, state_dict:dict[str, Tensor]):
    prefix = f"blk.{idx}."
    mtp_dict = {}
    for k in [k for k in state_dict if k.startswith(prefix)]:
      rel = k[len(prefix):]
      mtp_dict[rel[len("nextn."):] if rel.startswith("nextn.") else "block."+rel] = state_dict.pop(k)
    nn.state.load_state_dict(self, mtp_dict, consume=True, verbose=False)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv = Linear(config.dim, self.conv_channels, bias=False)
    if ssm.kda:
      self.ssm_g_a, self.ssm_g_b = Linear(config.dim, self.head_v_dim, bias=False), Linear(self.head_v_dim, ssm.inner_size, bias=False)
      self.ssm_f_a, self.ssm_f_b = Linear(config.dim, self.head_k_dim, bias=False), Linear(self.head_k_dim, ssm.inner_size, bias=False)
    else:
      self.attn_gate = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_alpha = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(ssm.inner_size if ssm.kda else self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads, 1) if ssm.kda else Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    # bind ints to a variable so the reset flag stays a runtime value (it toggles when generation restarts at position 0)
    start_pos = start_pos if isinstance(start_pos, UOp) else UOp.variable("start_pos", 0, self.config.max_context-1).bind(start_pos)
    initial = Tensor(start_pos).eq(0)
    is_kda = hasattr(self, "ssm_g_a")
    symbolic = isinstance(T, UOp)
    T_pad = x.max_shape[1]  # symbolic chunks are padded to their max size: one graph serves every size

    # input processing
    x = x.half()
    out_gate = self.ssm_g_b(self.ssm_g_a(x)) if is_kda else self.attn_gate(x)
    out_gate = out_gate.reshape(B, T, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, T, self.num_v_heads)
    alpha = self.ssm_f_b(self.ssm_f_a(x)) if is_kda else self.ssm_alpha(x)
    log_alpha = ((alpha.float() + self.ssm_dt["bias"]).softplus().reshape(B, T, self.num_v_heads, -1) *
                 self.ssm_a.reshape(self.num_v_heads, -1))

    # qkv conv, conv_state is reset when starting from position 0
    conv_state = initial.where(0, self.conv_state)
    # assemble the conv window in a static-size buffer: [conv_state | qkv rows | zero-pad].
    # padded steps are exact no-ops: beta=0 (delta rule off), log_alpha=0 (decay 1 after exp)
    win = Tensor.zeros(B, self.ssm_conv_kernel-1 + T_pad, self.conv_channels).uop
    win = win.after(win[:, :self.ssm_conv_kernel-1].store(conv_state.cast(win.dtype).uop))
    win = win.after(win[:, self.ssm_conv_kernel-1:self.ssm_conv_kernel-1+T].store(self.attn_qkv(x).cast(win.dtype).uop))
    conv_window = Tensor(win)
    # the last conv_kernel-1 columns of the window become the next conv state
    conv_state_store = self.conv_state.uop.store(conv_window[:, T:T+self.ssm_conv_kernel-1].cast(self.conv_state.dtype).uop)

    conv_out = functools.reduce(lambda a,b: a+b,
      (conv_window[:, i:i+T_pad] * self.ssm_conv1d["weight"][:, i] for i in range(self.ssm_conv_kernel))).silu()
    if symbolic:
      out_gate = out_gate.pad_to((B, T_pad, self.num_v_heads, self.head_v_dim))
      beta, log_alpha = beta.pad_to((B, T_pad, self.num_v_heads)), log_alpha.pad_to((B, T_pad, *log_alpha.shape[2:]))
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    qk_eps = 1e-12 if is_kda else 1e-6
    q, k = (z.reshape(B, T_pad, self.num_k_heads, self.head_k_dim).normalize(dim=-1, eps=qk_eps)
            .repeat(1, 1, self.num_v_heads//self.num_k_heads, 1) for z in (q, k))
    v = v.reshape(B, T_pad, self.num_v_heads, self.head_v_dim)
    # layout the per-step operands to broadcast against the (B, H, V, K) state
    q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
    q = q * self.head_k_dim**-0.5
    alpha = log_alpha.transpose(1, 2).exp()  # per-channel decay for kda, per-head otherwise (B, H, T, V|1)

    # recurrent: scan over the (padded) tokens, updating the recurrent state. collect the per-step outputs
    state = Tensor(self.recurrent_state.uop.after(conv_state_store))  # carry the conv write into this graph
    if self.head_k_dim % 32 == 0 and self.head_v_dim % 4 == 0 and amd_custom_kernels_supported(x.device):
      # one fused kernel for the whole scan; it resets and updates the recurrent state in place (RDNA3)
      core = gated_delta_prefill(q, k, v, beta, alpha, state, Tensor(start_pos)).transpose(1, 2)
    else:
      q, k, v, beta = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
      alpha = alpha.unsqueeze(-1)
      state = initial.where(0, state.float())
      outs = []
      for t in range(T_pad):
        s1 = state * alpha[:, :, t]  # decay the state
        delta = (v[:, :, t] - (s1*k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]  # the delta rule update
        state = s1 + delta * k[:, :, t]
        outs.append((state * q[:, :, t]).sum(-1))

      # store the updated recurrent state in place, then read the stacked outputs after the write
      state_store = self.recurrent_state.uop.store(state.cast(self.recurrent_state.dtype).uop)
      core = Tensor(outs[0].stack(*outs[1:], dim=1).contiguous().uop.after(state_store))

    # output; undo the padding before the output projection
    z = (self.ssm_norm(core) * (out_gate.sigmoid() if is_kda else out_gate.silu())).cast(x.dtype).contiguous()
    if symbolic: z = z[:, :T]
    return self.ssm_out(z.reshape(B, T, -1))

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      # fp16 state: halves VRAM (30 GDN blocks × 8 MB = 240 MB vs 480 MB fp32);
      # the scan kernel accumulates in fp32 registers and casts on store/load.
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim, dtype=dtypes.half, device=x.device).clone()

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = [GatedDeltaNetBlock(dense_config if i < config.leading_dense_blocks else config, config.ssm)
                               if config.ssm and config.ssm_layers[i] else
                               block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self.mtp: MTPBlock|None = None
    self.max_drafts = 8
    self.mtp_stats = {"accepted": 0, "total": 0}
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    return self._forward_hidden(tokens, start_pos, temperature)[0]

  def logits(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    """Raw next-token logits for the last token of ``tokens`` (no sampling).

    Used by external engines (e.g. FreeToken) that apply their own sampler.
    The Gumbel sample inside ``_forward_hidden`` is discarded; temperature is
    fixed at 1.0 so the logits are raw.
    """
    return self._forward_hidden(tokens, start_pos, Tensor([1.0]))[2]

  _per_layer_debug = False  # set on the instance to collect per-block stats (debug)

  def _forward_hidden(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    x = self.token_embd(tokens).float()                   # (B, T, D)
    if self._per_layer_debug: self._dbg_stats: list[Tensor] = []
    global dbg_stats, dbg_sub
    dbg_stats = self._dbg_stats if self._per_layer_debug else None
    dbg_sub = self._per_layer_debug and getattr(self, "_sub_stats", False)
    for i, block in enumerate(self.blk):
      x = block(x, start_pos)
      if self._per_layer_debug:
        # lazy per-layer stats (resolved in the caller's single realize): n_nan, n_inf, absmax
        self._dbg_stats.append(_sub_stat(x))
    # only run the output projection on the last token
    logits = self.output(self.output_norm(x[:, -1:]))[:, -1, :]
    if self._per_layer_debug: dbg_stats = None
    # Gumbel-sample trick: sample to softmax(logits/temp)
    sample = (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)
    return sample, x, logits

  def _logits_at(self, hidden:Tensor, idx:int) -> Tensor:
    return self.output(self.output_norm(hidden[:, idx:idx+1]))[:, 0, :]

  def mtp_forward(self, fused:Tensor, start_pos:int) -> Tensor:
    h_mtp = self.mtp.block(fused, start_pos)
    return self.output(self.mtp.shared_head_norm(h_mtp))[:, -1, :]

  def _set_mtp_flash(self, val: bool):
    for b in self.blk:
      if hasattr(b, "use_flash"): b.use_flash = val
    if self.mtp is not None and hasattr(self.mtp.block, "use_flash"):
      self.mtp.block.use_flash = val

  def _ensure_mtp_jits(self):
    if not hasattr(self, "mtp_draft_jit"):
      self.mtp_draft_jit = TinyJit(self.mtp_draft)
      self.mtp_verify_jit = TinyJit(self.mtp_verify)
      self.mtp_rollout_jit = TinyJit(self.mtp_rollout)
      self.mtp_verify_jits: dict[int, TinyJit] = {}  # per-T (K+1) exact-batch verify graphs

  def mtp_draft(self, tok:Tensor, h_q:Tensor, start_pos:int|UOp) -> tuple[Tensor, Tensor]:
    # draft the next token with the MTP head; also return the MTP block's hidden for chaining
    fused = self.mtp.fuse(self.token_embd(tok).float(), h_q)
    h_mtp = self.mtp.block(fused, start_pos)
    draft = self.output(self.mtp.shared_head_norm(h_mtp))[:, -1, :].argmax(-1, keepdim=True)
    return draft, h_mtp[:, -1:, :].clone()

  def mtp_verify(self, toks:Tensor, start_pos:int|UOp, temperature:Tensor) -> tuple[Tensor, Tensor]:
    # run the main model over `toks` (T>=2); return per-position sampled ids and the hidden
    x = self.token_embd(toks).float()
    for block in self.blk: x = block(x, start_pos)
    logits_all = self.output(self.output_norm(x))
    # gumbel noise at the concrete max shape, sliced to the actual (symbolic) T
    noise = Tensor.rand(1, MTP_TMAX, logits_all.shape[-1])[:, :logits_all.shape[1]]
    sample_ids = (logits_all / temperature.maximum(1e-12) - (noise.maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)
    return sample_ids, x.contiguous()

  def mtp_verify_fixed(self, toks:Tensor, start_pos:int|UOp, temperature:Tensor) -> tuple[Tensor, Tensor]:
    """Exact-batch verify: run the main model over a *constant* T=K+1 token batch (no
    32-padding). Must be captured with use_flash=False so the full-attention blocks use
    standard (exact-T) attention instead of the padded batched flash; the GatedDelta
    recurrent scan then runs exactly T steps (T_pad == T)."""
    x = self.token_embd(toks).float()
    for block in self.blk: x = block(x, start_pos)
    logits_all = self.output(self.output_norm(x))
    noise = Tensor.rand(1, x.shape[1], logits_all.shape[-1])
    sample_ids = (logits_all / temperature.maximum(1e-12) - (noise.maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)
    return sample_ids, x.contiguous()

  def _get_mtp_verify_jit(self, T:int) -> TinyJit:
    # one exact-batch TinyJit per T (K+1); recompiles lazily only when the draft count changes
    if T not in self.mtp_verify_jits:
      self.mtp_verify_jits[T] = TinyJit(self.mtp_verify_fixed)
    return self.mtp_verify_jits[T]

  def mtp_rollout(self, tok:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    # T=1 main forward returning the hidden of `tok` at `start_pos` (reject path / first queued)
    _, h, _ = self._forward_hidden(tok, start_pos, temperature)
    return h.clone()

  def _mtp_warmup(self):
    self._ensure_mtp_jits()
    try:
      sp = UOp.variable("start_pos", 0, self.max_context-1)
      temp = Tensor([0.0])
      dim = self.token_embd.weight.shape[1]
      h0 = Tensor.zeros(1, 1, dim)
      # draft (T=1 MTP block) + rollout with flash (T=1 decode kernels)
      self._set_mtp_flash(True)
      d, _ = self.mtp_draft_jit(Tensor([[0]], dtype="int32"), h0, sp.bind(0)); d.realize()
      self.mtp_rollout_jit(Tensor([[0]], dtype="int32"), sp.bind(0), temp).realize()
      # exact-batch verify (T=2) with standard attention
      self._set_mtp_flash(False)
      s, h = self._get_mtp_verify_jit(2)(Tensor([[0, 0]], dtype="int32"), sp.bind(0), temp)
      s.realize(); h.realize()
    finally:
      self._set_mtp_flash(True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, state_dict = gguf_load(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    ssm_layers: tuple[bool, ...] = ()
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
      ssm_layers = tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(kv[f'{arch}.block_count']))
    elif arch == 'kimi-linear':
      ssm_layers = tuple(x == 0 for x in n_kv_heads)
      n_kv_heads = max(n_kv_heads)
      ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.kda.head_dim'], n_heads, n_heads, n_heads*kv[f'{arch}.kda.head_dim'], kda=True)
      for i, is_ssm in enumerate(ssm_layers):
        if not is_ssm: continue
        state_dict[f"blk.{i}.attn_qkv.weight"] = state_dict.pop(f"blk.{i}.attn_q.weight").cat(
          state_dict.pop(f"blk.{i}.attn_k.weight"), state_dict.pop(f"blk.{i}.attn_v.weight"), dim=0).contiguous()
        state_dict[f"blk.{i}.ssm_conv1d.weight"] = state_dict.pop(f"blk.{i}.ssm_conv1d_q.weight").cat(
          state_dict.pop(f"blk.{i}.ssm_conv1d_k.weight"), state_dict.pop(f"blk.{i}.ssm_conv1d_v.weight"), dim=0).squeeze(1).contiguous()
        state_dict[f"blk.{i}.ssm_out.weight"] = state_dict.pop(f"blk.{i}.attn_output.weight")
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if arch == 'kimi-linear': continue
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch == 'llama' and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
    config = TransformerConfig(
      num_blocks=kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0), dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv[f'{arch}.rope.freq_base'],
      rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe', 'kimi-linear')),
      expert_gating_func=ExpertGating(kv.get(f'{arch}.expert_gating_func', ExpertGating.SOFTMAX)),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict)
    model = Transformer(config)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # MTP head: hold `blk.<num_blocks>` as a dedicated draft block instead of dropping it
    if (nextn := kv.get(f'{arch}.nextn_predict_layers', 0)) > 0:
      if nextn > config.num_blocks: raise ValueError(f"nextn_predict_layers={nextn} > block_count - nextn ({config.num_blocks})")
      block_cls = MLATransformerBlock if kv_lora_rank > 0 else TransformerBlock
      model.mtp = MTPBlock(config, block_cls)
      model.mtp.load_from_gguf(config.num_blocks, state_dict)
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def warmup(self):
    for _ in range(2): list(zip(range(2), self.generate([0])))
    if self.mtp is not None: self._mtp_warmup()

  def get_start_pos(self, tokens:list[int]) -> int:
    # recurrent state can't be partially reused after divergence: reuse it only when tokens extend the cached prefix
    if self.has_recurrent_block:
      return len(self._cached_tokens) if self._cached_tokens and len(self._cached_tokens) < len(tokens) \
        and tokens[:len(self._cached_tokens)] == self._cached_tokens else 0
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def get_mtp_start_pos(self, tokens:list[int]) -> int:
    # MTP reuses both the main and MTP-block caches only when `tokens` extends the cached prefix
    if not self._cached_tokens or len(self._cached_tokens) >= len(tokens): return 0
    return len(self._cached_tokens) if tokens[:len(self._cached_tokens)] == self._cached_tokens else 0

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0, mtp:bool=False, num_drafts:int=1, adaptive:bool=False):
    if mtp and self.mtp is not None and num_drafts > 0:
      yield from self._generate_mtp(tokens, temperature, num_drafts, adaptive)
      return
    if self.has_recurrent_block and not amd_custom_kernels_supported(self.token_embd.weight.device): chunk_size = 1
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor([temperature])
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      n_toks = min(chunk_size, len(tokens) - start_pos)
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()
      start_pos += n_toks
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]

  def _generate_mtp(self, tokens:list[int], temperature:float, num_drafts:int=1, adaptive:bool=False):
    """JIT'd MTP speculative decoding. On AMD the batched flash kernel needs the
    query length to be a multiple of 32, but the MTP verify/prefill use small
    variable T, so they fall back to the standard attention path. See `_mtp_loop`
    for the decoder body."""
    self._ensure_mtp_jits()
    try:
      yield from self._mtp_loop(tokens, temperature, num_drafts, adaptive)
    finally:
      self._set_mtp_flash(True)

  def _mtp_loop(self, tokens:list[int], temperature:float, num_drafts:int=1, adaptive:bool=False):
    """MTP speculative decoding: drafts `num_drafts` tokens with the MTP head,
    verifies them against the main model in one batched forward, and emits the
    longest accepted prefix. Uses three JIT graphs: mtp_draft (T=1 MTP block),
    mtp_verify (variable-T main, T>=2), and mtp_rollout (T=1 main, reject path).
    Maintains a separate MTP-block KV cache over the fused prompt history.
    Does not mutate the caller's `tokens` list; builds a fresh stream instead."""
    num_drafts = max(1, min(num_drafts, self.max_drafts))
    window = []
    stream = list(tokens)
    max_ctx, prompt_len = self.max_context, len(stream)
    if prompt_len >= max_ctx: return
    if prompt_len == 0: raise ValueError("MTP generate requires at least one token")
    temp = Tensor([temperature])
    int_dtype = "int32"
    sp = UOp.variable("start_pos", 0, max_ctx-1)
    # 1) prefill (1 or the re-used tail). use standard attention here so any prompt
    #    length works (the batched flash kernel requires T to be a multiple of 32).
    self._set_mtp_flash(False)
    start = self.get_mtp_start_pos(stream)
    if start == 0:
      pt = Tensor(stream, dtype=int_dtype).reshape(1, -1)
      first_tok, hidden_p, _ = self._forward_hidden(pt, 0, temp)
      self.mtp.block(self.mtp.fuse(self.token_embd(pt).float(), hidden_p), 0)
    else:
      tail = stream[start:]
      pt = Tensor(tail, dtype=int_dtype).reshape(1, -1)
      first_tok, hidden_p, _ = self._forward_hidden(pt, start, temp)
      self.mtp.block(self.mtp.fuse(self.token_embd(pt).float(), hidden_p), start)
    first_tok = int(first_tok.item())
    # enable flash for the hot loop draft/rollout (T=1 decode); the exact-batch verify
    # toggles it off (standard attention) so the GatedDelta scan and attention run T=K+1.
    self._set_mtp_flash(True)
    # 2) queue the first generated token and compute its hidden via a T=1 rollout
    queued, queued_pos = first_tok, prompt_len
    yield first_tok
    stream.append(first_tok)
    self._cached_tokens = stream[:-1]
    h_q = self.mtp_rollout_jit(Tensor([[queued]], dtype=int_dtype), sp.bind(queued_pos), temp).realize()
    while queued_pos + 1 < max_ctx:
      K = min(num_drafts, max_ctx - queued_pos - 1)
      # draft K tokens autoregressively with the MTP head
      drafts = []
      h_d = h_q
      tok = queued
      for i in range(K):
        b, h_mtp = self.mtp_draft_jit(Tensor([[tok]], dtype=int_dtype), h_d, sp.bind(queued_pos+i))
        b = int(b.realize().item())
        drafts.append(b)
        tok = b
        h_d = h_mtp
      # verify [queued] + drafts in one exact-batch main forward (standard attention).
      # the fixed T=K+1 shape makes T_pad == T, so the GatedDelta scan is exact.
      verify_toks = [queued] + drafts
      T = len(verify_toks)
      self._set_mtp_flash(False)
      sample_ids, h_all = self._get_mtp_verify_jit(T)(Tensor([verify_toks], dtype=int_dtype), sp.bind(queued_pos), temp)
      self._set_mtp_flash(True)
      sample_ids, h_all = sample_ids.realize(), h_all.realize()
      # longest-prefix accept: find the first draft the main model rejects
      a = 0
      while a < K and drafts[a] == int(sample_ids[:, a].item()): a += 1
      self.mtp_stats["total"] += K
      self.mtp_stats["accepted"] += a
      if adaptive:
        window.append(a / K)
        if len(window) > 64: window.pop(0)
        acc = sum(window) / len(window)
        if acc > 0.7 and num_drafts < self.max_drafts: num_drafts += 1
        elif acc < 0.4 and num_drafts > 1: num_drafts -= 1
      if a == K:
        # full accept: emit all K drafts, continue from the last draft
        for b in drafts:
          stream.append(b); self._cached_tokens = stream[:-1]; yield b
        h_q = h_all[:, K:K+1, :].clone()
        queued = drafts[-1]
        queued_pos += K
      else:
        # partial/full reject: emit the accepted prefix + the true token at position a
        for b in drafts[:a]:
          stream.append(b); self._cached_tokens = stream[:-1]; yield b
        true_tok = int(sample_ids[:, a].item())
        stream.append(true_tok); self._cached_tokens = stream[:-1]; yield true_tok
        h_q = self.mtp_rollout_jit(Tensor([[true_tok]], dtype=int_dtype), sp.bind(queued_pos+a+1), temp).realize()
        queued = true_tok
        queued_pos += a + 1
