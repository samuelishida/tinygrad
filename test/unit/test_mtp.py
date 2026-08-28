import struct, tempfile, pathlib, unittest
import numpy as np
from tinygrad import Tensor, UOp
from tinygrad.llm.model import Transformer, TransformerConfig, TransformerBlock, MTPBlock
from tinygrad.nn.state import get_state_dict, load_state_dict

DIM, HIDDEN, N_HEADS, VOCAB, CTX = 16, 32, 2, 32, 32

def _config(num_blocks=4, max_context=CTX):
  return TransformerConfig(num_blocks=num_blocks, dim=DIM, hidden_dim=HIDDEN, n_heads=N_HEADS,
    n_kv_heads=N_HEADS, norm_eps=1e-5, vocab_size=VOCAB, head_dim=DIM//N_HEADS, rope_theta=10000,
    rope_dim=DIM//N_HEADS, v_head_dim=DIM//N_HEADS, max_context=max_context)

def _make_model(with_mtp=True, num_blocks=4, max_context=CTX):
  c = _config(num_blocks, max_context)
  m = Transformer(c)
  if with_mtp: m.mtp = MTPBlock(c, TransformerBlock)
  sd = get_state_dict(m)
  for k in sd: sd[k] = Tensor.uniform(*sd[k].shape, low=-0.1, high=0.1)
  load_state_dict(m, sd, verbose=False)
  return m

def _ref_generate_mtp(m, tokens, temperature):
  # non-JIT reference mirroring the pre-JIT _generate_mtp (greedy at temp=0)
  stream = list(tokens)
  max_ctx, prompt_len = m.max_context, len(stream)
  if prompt_len >= max_ctx: return
  temp = Tensor([temperature])
  int_dtype = "int32"
  pt = Tensor(stream, dtype=int_dtype).reshape(1, -1)
  first_tok, hidden_p, _ = m._forward_hidden(pt, 0, temp)
  first_tok = int(first_tok.item())
  m.mtp.block(m.mtp.fuse(m.token_embd(pt).float(), hidden_p), 0)
  queued, queued_pos = first_tok, prompt_len
  yield first_tok
  stream.append(first_tok)
  m._cached_tokens = stream[:-1]
  _, hidden_q, _ = m._forward_hidden(Tensor([[queued]], dtype=int_dtype), queued_pos, temp)
  while queued_pos + 1 < max_ctx:
    fused_q = m.mtp.fuse(m.token_embd(Tensor([[queued]], dtype=int_dtype)).float(), hidden_q)
    b = int(m.output(m.mtp.shared_head_norm(m.mtp.block(fused_q, queued_pos)))[:, -1, :].argmax(-1).item())
    _, hidden_pair, _ = m._forward_hidden(Tensor([[queued, b]], dtype=int_dtype), queued_pos, temp)
    verify_logits = m._logits_at(hidden_pair, 0)
    a2 = int((verify_logits / temp.maximum(1e-12) - (Tensor.rand_like(verify_logits).maximum(1e-12).log().neg()).log()).argmax(-1).item())
    if b == a2:
      queued_hidden = hidden_pair[:, -1:, :]
      queued = b
      stream.append(b); m._cached_tokens = stream[:-1]; yield b
    else:
      stream.append(a2); m._cached_tokens = stream[:-1]; yield a2
      _, hidden_q, _ = m._forward_hidden(Tensor([[a2]], dtype=int_dtype), queued_pos+1, temp)
      queued, queued_hidden = a2, hidden_q
    queued_pos += 1
    hidden_q = queued_hidden

class TestMTPBlockLoad(unittest.TestCase):
  def test_load_from_gguf_renames_and_pops(self):
    idx = 5
    mtp = MTPBlock(_config(), TransformerBlock)
    gguf_sd = {}
    for k, v in get_state_dict(mtp).items():
      if k.startswith("block."):
        gguf_sd[f"blk.{idx}.{k[len('block.'):]}"] = Tensor.ones(*v.shape)
      else:
        gguf_sd[f"blk.{idx}.nextn.{k}"] = Tensor.ones(*v.shape)
    fresh = MTPBlock(_config(), TransformerBlock)
    fresh.load_from_gguf(idx, gguf_sd)
    self.assertEqual(len(gguf_sd), 0, "all MTP keys must be popped from state_dict")
    for v in get_state_dict(fresh).values():
      np.testing.assert_array_equal(v.numpy(), np.ones(v.shape))

class TestForwardHidden(unittest.TestCase):
  def test_forward_hidden_returns_tuple(self):
    m = _make_model()
    sample, hidden, logits = m._forward_hidden(Tensor([[0, 1, 2]]), 3, Tensor([0.0]))
    self.assertEqual(sample.shape, (1, 1))
    self.assertEqual(hidden.shape, (1, 3, DIM))
    self.assertEqual(logits.shape, (1, VOCAB))
    s = m.forward(Tensor([[0, 1, 2]]), 3, Tensor([0.0]))
    self.assertEqual(s.shape, (1, 1))

class TestMTPForward(unittest.TestCase):
  def test_mtp_forward_shape(self):
    m = _make_model()
    logits = m.mtp_forward(Tensor.rand(1, 1, DIM), 0)
    self.assertEqual(logits.shape, (1, VOCAB))
    fused2 = m.mtp.fuse(Tensor.rand(1, 1, DIM), Tensor.rand(1, 1, DIM))
    self.assertEqual(fused2.shape, (1, 1, DIM))

class TestGenerateMTP(unittest.TestCase):
  def test_generate_mtp_runs_full_context(self):
    m = _make_model(max_context=CTX)
    toks = list(m.generate([3, 7, 2, 9], temperature=0.0, mtp=True))
    self.assertEqual(len(toks), CTX - 4)
    self.assertTrue(all(isinstance(t, int) for t in toks))

  def test_mtp_falls_back_when_absent(self):
    m = _make_model(with_mtp=False)
    prompt = [3, 7]
    fast = list(m.generate(prompt[:], temperature=0.0))
    mtp = list(m.generate(prompt[:], temperature=0.0, mtp=True))
    self.assertEqual(mtp, fast)

  def test_mtp_temperature0_is_deterministic(self):
    m = _make_model()
    prompt = [1, 2, 3, 4]
    self.assertEqual(list(m.generate(prompt[:], temperature=0.0, mtp=True)),
                     list(m.generate(prompt[:], temperature=0.0, mtp=True)))

  def test_generate_mtp_jit_equals_reference(self):
    m = _make_model()
    jit = list(m.generate([1, 2, 3, 4], temperature=0.0, mtp=True))
    m2 = _make_model()
    load_state_dict(m2, get_state_dict(m), verbose=False)
    ref = list(_ref_generate_mtp(m2, [1, 2, 3, 4], 0.0))
    self.assertEqual(jit, ref)

  def test_mtp_verify_shape(self):
    m = _make_model()
    m._ensure_mtp_jits()
    sp = UOp.variable("start_pos", 0, m.max_context-1)
    vt = UOp.variable("toks", 1, m.max_drafts+1)
    temp = Tensor([0.0])
    t2 = Tensor([[0, 1]], dtype="int32")
    s2, h2 = m.mtp_verify_jit(t2[:, :vt.bind(2)], sp.bind(0), temp)
    s2, h2 = s2.realize(), h2.realize()
    self.assertEqual(h2[:, :2].numpy().shape, (1, 2, DIM))
    self.assertEqual(s2[:, :2].numpy().shape, (1, 2, 1))

  def test_mtp_verify_fixed_matches_variable(self):
    # the exact-batch verify must produce the same (greedy, temp=0) sample ids as the
    # variable-T verify for the same token batch
    m = _make_model()
    m._ensure_mtp_jits()
    sp = UOp.variable("start_pos", 0, m.max_context-1)
    vt = UOp.variable("toks", 1, m.max_drafts+1)
    temp = Tensor([0.0])
    toks = [2, 5, 8]
    sf, hf = m._get_mtp_verify_jit(3)(Tensor([toks], dtype="int32"), sp.bind(0), temp)
    sf, hf = sf.realize(), hf.realize()
    tvar = Tensor([toks + [0]*(m.max_drafts+1 - 3)], dtype="int32")
    sv, hv = m.mtp_verify_jit(tvar[:, :vt.bind(3)], sp.bind(0), temp)
    sv, hv = sv.realize(), hv.realize()
    np.testing.assert_array_equal(sf[:, :3].numpy(), sv[:, :3].numpy())
    self.assertEqual(hf[:, :3].numpy().shape, (1, 3, DIM))

  def test_mtp_verify_jit_cached_per_t(self):
    m = _make_model()
    m._ensure_mtp_jits()
    self.assertIs(m._get_mtp_verify_jit(3), m._get_mtp_verify_jit(3))
    self.assertIsNot(m._get_mtp_verify_jit(3), m._get_mtp_verify_jit(5))

  def test_carried_hidden_is_real_buffer(self):
    m = _make_model()
    m._ensure_mtp_jits()
    sp = UOp.variable("start_pos", 0, m.max_context-1)
    temp = Tensor([0.0])
    h = m.mtp_rollout_jit(Tensor([[0]], dtype="int32"), sp.bind(0), temp).realize()
    d, _ = m.mtp_draft_jit(Tensor([[0]], dtype="int32"), h, sp.bind(0))
    d.realize()

  def test_warmup_compiles_mtp_jits(self):
    m = _make_model()
    m.warmup()
    self.assertTrue(hasattr(m, "mtp_draft_jit"))
    self.assertTrue(hasattr(m, "mtp_verify_jit"))
    self.assertTrue(hasattr(m, "mtp_rollout_jit"))

  def test_mtp_multi_draft_runs_full_context(self):
    m = _make_model(max_context=CTX)
    toks = list(m.generate([3, 7, 2, 9], temperature=0.0, mtp=True, num_drafts=4))
    self.assertEqual(len(toks), CTX - 4)
    self.assertTrue(all(0 <= t < VOCAB for t in toks))

  def test_mtp_multi_draft_deterministic(self):
    m = _make_model()
    prompt = [1, 2, 3, 4]
    a = list(m.generate(prompt[:], temperature=0.0, mtp=True, num_drafts=4))
    b = list(m.generate(prompt[:], temperature=0.0, mtp=True, num_drafts=4))
    self.assertEqual(a, b)

  def test_mtp_multi_draft_various_k(self):
    for k in (1, 2, 4, 8):
      m = _make_model(max_context=CTX)
      toks = list(m.generate([3, 7, 2, 9], temperature=0.0, mtp=True, num_drafts=k))
      self.assertEqual(len(toks), CTX - 4, f"num_drafts={k}")

  def test_mtp_multi_draft_equals_single(self):
    # num_drafts=1 must match the single-draft reference
    m = _make_model()
    single = list(m.generate([1, 2, 3, 4], temperature=0.0, mtp=True, num_drafts=1))
    m2 = _make_model()
    load_state_dict(m2, get_state_dict(m), verbose=False)
    ref = list(_ref_generate_mtp(m2, [1, 2, 3, 4], 0.0))
    self.assertEqual(single, ref)

  def test_get_mtp_start_pos(self):
    m = _make_model()
    m._cached_tokens = [1, 2, 3, 4, 5]
    self.assertEqual(m.get_mtp_start_pos([1, 2, 3, 4, 5, 6]), 5)
    self.assertEqual(m.get_mtp_start_pos([1, 2, 3, 4, 5, 6, 7]), 5)
    self.assertEqual(m.get_mtp_start_pos([1, 2, 3, 4, 5]), 0)  # not longer
    self.assertEqual(m.get_mtp_start_pos([9, 8, 7]), 0)  # divergent
    self.assertEqual(m.get_mtp_start_pos([]), 0)  # empty

  def test_mtp_multi_turn_reuse(self):
    m = _make_model()
    prompt1 = [1, 2, 3, 4]
    gen1 = list(m.generate(prompt1[:], temperature=0.0, mtp=True))
    # turn 2 extends the cached prefix (as serve/cli re-encode the full history)
    prompt2 = prompt1 + gen1 + [5, 6]
    gen2 = list(m.generate(prompt2[:], temperature=0.0, mtp=True))
    m2 = _make_model()
    load_state_dict(m2, get_state_dict(m), verbose=False)
    gen2_cold = list(m2.generate(prompt2[:], temperature=0.0, mtp=True))
    self.assertEqual(gen2, gen2_cold)

  def test_mtp_multi_turn_divergent_reset(self):
    m = _make_model()
    list(m.generate([1, 2, 3, 4], temperature=0.0, mtp=True))
    prompt2 = [9, 8, 7]
    gen2 = list(m.generate(prompt2[:], temperature=0.0, mtp=True))
    m2 = _make_model()
    load_state_dict(m2, get_state_dict(m), verbose=False)
    gen2_cold = list(m2.generate(prompt2[:], temperature=0.0, mtp=True))
    self.assertEqual(gen2, gen2_cold)

  def test_mtp_adaptive_runs(self):
    m = _make_model(max_context=CTX)
    toks = list(m.generate([3, 7, 2, 9], temperature=0.0, mtp=True, num_drafts=1, adaptive=True))
    self.assertEqual(len(toks), CTX - 4)

  def test_mtp_stats_monotonic(self):
    m = _make_model(max_context=CTX)
    list(m.generate([3, 7, 2, 9], temperature=0.0, mtp=True, num_drafts=4))
    s = m.mtp_stats
    self.assertGreater(s["total"], 0)
    self.assertGreaterEqual(s["accepted"], 0)
    self.assertLessEqual(s["accepted"], s["total"])

  def test_mtp_empty_prompt_rejected(self):
    m = _make_model()
    with self.assertRaises(ValueError):
      list(m.generate([], temperature=0.0, mtp=True))

  def test_mtp_single_token_context(self):
    m = _make_model(max_context=5)
    toks = list(m.generate([1, 2, 3, 4], temperature=0.0, mtp=True))
    self.assertEqual(len(toks), 1)  # only the first generated token

  def test_mtp_num_drafts_zero_fast_path(self):
    m = _make_model()
    prompt = [3, 7]
    fast = list(m.generate(prompt[:], temperature=0.0))
    mtp0 = list(m.generate(prompt[:], temperature=0.0, mtp=True, num_drafts=0))
    self.assertEqual(mtp0, fast)

  def test_mtp_temperature_positive_deterministic(self):
    m1 = _make_model()
    m2 = _make_model()
    load_state_dict(m2, get_state_dict(m1), verbose=False)
    Tensor.manual_seed(42)
    a = list(m1.generate([1, 2, 3, 4], temperature=0.7, mtp=True))
    Tensor.manual_seed(42)
    b = list(m2.generate([1, 2, 3, 4], temperature=0.7, mtp=True))
    self.assertEqual(a, b)

# ---- tiny GGUF with an MTP head (mistral arch: no reshape, no ssm) ----

def _enc_val(buf, v):
  if isinstance(v, str): buf += struct.pack("<i", 8) + struct.pack("<Q", len(v)) + v.encode()
  elif isinstance(v, int): buf += struct.pack("<i", 4) + struct.pack("<I", v)
  elif isinstance(v, float): buf += struct.pack("<i", 6) + struct.pack("<f", v)
  elif isinstance(v, list):
    buf += struct.pack("<i", 9) + struct.pack("<i", 8) + struct.pack("<Q", len(v))
    for s in v: sb = s.encode(); buf += struct.pack("<Q", len(sb)) + sb
  else: raise TypeError(v)

def _build_tiny_gguf(path, with_mtp=True):
  dim, n_heads, n_kv, hidden, vocab_size = 16, 2, 2, 32, 16
  head_dim = dim // n_heads
  block_count, nextn = (2, 1) if with_mtp else (1, 0)
  n_main = block_count - nextn
  tensors = []
  def add(name, dims):
    n = 1
    for d in dims: n *= d
    tensors.append((name, tuple(dims), 0, np.full(n, 0.1, dtype=np.float32).tobytes()))
  add("token_embd.weight", (vocab_size, dim))
  add("output.weight", (vocab_size, dim))
  add("output_norm.weight", (dim,))
  for i in range(block_count):
    add(f"blk.{i}.attn_norm.weight", (dim,))
    add(f"blk.{i}.attn_q.weight", (head_dim*n_heads, dim))
    add(f"blk.{i}.attn_k.weight", (head_dim*n_kv, dim))
    add(f"blk.{i}.attn_v.weight", (head_dim*n_kv, dim))
    add(f"blk.{i}.attn_output.weight", (dim, head_dim*n_heads))
    add(f"blk.{i}.ffn_norm.weight", (dim,))
    add(f"blk.{i}.ffn_gate.weight", (hidden, dim))
    add(f"blk.{i}.ffn_up.weight", (hidden, dim))
    add(f"blk.{i}.ffn_down.weight", (dim, hidden))
  if with_mtp:
    i = n_main
    add(f"blk.{i}.nextn.eh_proj.weight", (dim, 2*dim))
    add(f"blk.{i}.nextn.enorm.weight", (dim,))
    add(f"blk.{i}.nextn.hnorm.weight", (dim,))
    add(f"blk.{i}.nextn.shared_head_norm.weight", (dim,))
  kvs = [("general.architecture", "mistral"), ("mistral.context_length", 256),
         ("mistral.embedding_length", dim), ("mistral.feed_forward_length", hidden),
         ("mistral.attention.head_count", n_heads), ("mistral.attention.head_count_kv", n_kv),
         ("mistral.attention.layer_norm_rms_epsilon", 1e-5), ("mistral.rope.freq_base", 10000),
         ("mistral.block_count", block_count), ("tokenizer.ggml.tokens", [f"tok{i}" for i in range(vocab_size)])]
  if with_mtp: kvs.insert(9, ("mistral.nextn_predict_layers", nextn))
  buf = bytearray()
  buf += struct.pack("<4siqq", b"GGUF", 3, len(tensors), len(kvs))
  for k, v in kvs:
    kb = k.encode(); buf += struct.pack("<Q", len(kb)) + kb; _enc_val(buf, v)
  data_off = 0
  for name, dims, qtype, data in tensors:
    nb = name.encode()
    buf += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims))
    for d in reversed(dims): buf += struct.pack("<Q", d)
    buf += struct.pack("<i", qtype) + struct.pack("<Q", data_off)
    data_off += len(data)
  buf += b"\x00" * ((32 - len(buf) % 32) % 32)
  for _, _, _, data in tensors: buf += data
  path.write_bytes(bytes(buf))

class TestFromGgufMTP(unittest.TestCase):
  def test_from_gguf_builds_mtp_block(self):
    with tempfile.TemporaryDirectory() as d:
      p = pathlib.Path(d) / "tiny_mtp.gguf"
      _build_tiny_gguf(p, with_mtp=True)
      model, kv = Transformer.from_gguf(p, max_context=256, realize=False)
      self.assertIsNotNone(model.mtp)
      self.assertEqual(len(model.blk), 1)  # 2 blocks - 1 nextn = 1 main block
      self.assertEqual(model.mtp.eh_proj.weight.shape, (DIM, 2*DIM))
      self.assertEqual(model.mtp.enorm.weight.shape, (DIM,))
      self.assertEqual(kv["mistral.nextn_predict_layers"], 1)

  def test_from_gguf_non_mtp_has_no_mtp(self):
    with tempfile.TemporaryDirectory() as d:
      p = pathlib.Path(d) / "tiny_nomtp.gguf"
      _build_tiny_gguf(p, with_mtp=False)
      model, _ = Transformer.from_gguf(p, max_context=256, realize=False)
      self.assertIsNone(model.mtp)
      self.assertEqual(len(model.blk), 1)  # single main block, no MTP

if __name__ == "__main__":
  unittest.main()
