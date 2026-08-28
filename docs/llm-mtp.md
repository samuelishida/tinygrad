# MTP speculative decoding in tinygrad

`tinygrad/llm/model.py` loads and runs the **multi-token-prediction (MTP) head**
that some GGUFs ship with (DeepSeek-V3 / Qwen3-Next style, e.g.
`qwen35.nextn_predict_layers = 1`). Native Ollama uses this to reach higher
decode throughput on the same weights — tinygrad previously **stripped** the MTP
block (`num_blocks = block_count - nextn_predict_layers`).

This page documents the JIT'd, multi-draft, multi-turn MTP implementation, how
to enable it, and its current limitations.

## Why it matters

On `smtek/Qwen3.8-27B` (qwen35 arch, RX 7900 XTX / gfx1100), Ollama decodes at
~50.7 tok/s while the single-token tinygrad fast path does ~39.3 tok/s on the
same model/GPU (~29% gap). Both hit ~580 GB/s memory bandwidth, so the gap is not
raw GEMM — it is MTP: Ollama drafts a token with the cheap MTP head and verifies
it in the same forward, emitting up to 2 tokens per main-model forward pass.

## How it works

### Loading

`Transformer.from_gguf` no longer drops the MTP block. When the GGUF metadata
has `nextn_predict_layers > 0`, it builds a dedicated `MTPBlock`:

- keeps main blocks as `blk.0 .. blk.{block_count-nextn-1}`
- loads the leftover `blk.{block_count-nextn}.*` block weights and
  `blk.{block_count-nextn}.nextn.*` fusion weights into `model.mtp`
- uses the same transformer block class as the main blocks (`TransformerBlock`
  for qwen35, since `attention.kv_lora_rank` is absent)

The MTP block fuses the current token embedding and the previous-position
hidden state:

```
e      = enorm(emb)
h      = hnorm(hidden)
fused  = eh_proj(cat(e, h))
h_mtp  = block(fused)                       # dedicated transformer block
logits = output(shared_head_norm(h_mtp))    # shared output head
```

### JIT'd hot loop

The MTP decoder uses three `TinyJit` graphs:

- `mtp_draft_jit` — T=1 MTP block: drafts the next token and returns the MTP
  block's hidden for chaining.
- `mtp_verify_fixed` — **exact-batch** main model (`T = K+1`): runs
  `[queued] + drafts` in one forward and returns per-position sampled ids and
  the hidden. One graph is cached per draft count (`mtp_verify_jits[T]`). It is
  captured with `use_flash=False`, so the full-attention blocks use standard
  (exact-T) attention and the GatedDelta recurrent scan runs exactly `T` steps —
  no 32-padding (AMD batched flash only supports query lengths that are a
  multiple of 32, which would waste ~6× work on a 5-token verify).

- `mtp_rollout_jit` — T=1 main forward returning a position's hidden (the first
  queued / reject path).

The fast path and its JITs are unchanged (bit-for-bit).

### Multi-draft speculative decoding

`generate(..., mtp=True, num_drafts=K)`:

1. prefill the main model over the prompt (or the new tail on multi-turn reuse);
2. build the MTP-block KV over the fused prompt history;
3. draft K tokens autoregressively with the MTP head (each using the previous
   MTP hidden as the fusion input);
4. verify `[queued] + drafts` in one exact-batch (`T=K+1`) main forward;
5. accept the **longest prefix** of drafts that matches the main model's
   predictions, emitting up to K+1 tokens on a full hit.

Rejected-draft KV positions are safe: the causal attention mask
(`triu(start_pos+1)`) masks out future columns, so the next batch starts at the
first rejected position and overwrites stale entries before attending to them.

### Multi-turn KV reuse

Both the main-model and MTP-block caches persist across `generate` calls.
`serve.py` and `cli.py` re-render/re-encode the full accumulated message history
each turn, so the new prompt extends the previous `_cached_tokens` prefix.
`get_mtp_start_pos` returns the reusable prefix length; only the newly-appended
tail is prefilled (main model) and fused (MTP block). A divergent prompt falls
back to a full reset.

### Acceptance-rate observability

`Transformer.mtp_stats` tracks `{"accepted", "total"}` drafts. The CLI reports
`mtp accept N/M (P%)` after a benchmark. `--mtp-drafts auto` adjusts K with a
rolling-window hysteresis (raise while acceptance > 0.7, drop while < 0.4),
clamped to [1, 8]. Each distinct K uses its own exact-batch verify graph, so a
K change lazily recompiles one `T=K+1` graph.

## CLI / server

```
tinygrad.llm.cli --model smtek/Qwen3.8-27B:latest --mtp --benchmark 8
tinygrad.llm.cli --model smtek/Qwen3.8-27B:latest --mtp --mtp-drafts 4 --benchmark 8
tinygrad.llm.cli --model smtek/Qwen3.8-27B:latest --mtp --mtp-drafts auto --serve
```

- `--mtp` enables MTP (opt-in, defaults off).
- `--mtp-drafts` sets the draft count (int, default 1) or `auto` for adaptive.
- `--mtp` with a model that has no MTP head prints a warning and falls back to
  the fast path.

## Measurement methodology

MTP tok/s must be measured as **total emitted tokens / wall time**, not
per-yield, because the generator may emit multiple tokens between verify calls.
The `--benchmark` loop calls `next(gen)` once per count; for a fair MTP
comparison, drain a fixed number of tokens and divide by elapsed time.

## AMD results (RX 7900 XTX / gfx1100, smtek/Qwen3.8-27B)

Measured as total emitted tokens / wall time (temperature 0):

| mode | tok/s |
|------|-------|
| fast path (no MTP) | ~33–39 |
| MTP K=1 (exact T=2) | 15.2 |
| MTP K=4 (exact T=5) | 23.9 |
| Ollama reference | ~50.7 |

An exact-batch verify (this doc) roughly **halves** the old padded cost and is
~2.5× faster at K=1 vs the old variable-T path. But on this hybrid GPU the
64-block forward is *compute*-bound per verified position (a T=5 verify is
~205 ms vs a ~25 ms single weight-load), so batching reads weights once yet still
pays ~5× the per-position compute; the batched verify therefore can't beat the
fast path. A custom small-M attention kernel would not close this gap (the
residual cost is the model's per-position compute, not attention). On memory-
bound (datacenter-class) GPUs this cost structure differs and MTP can win.

## Current limitations

- **One MTP head.** `nextn_predict_layers = 1` yields a single MTP block, so
  multi-draft is achieved by autoregressive chaining (K runs of the one head),
  not by multiple heads in one forward.
- **Non-JIT prefill.** The variable-length prompt prefill is a single non-JIT
  forward per turn (one compile per session); the hot loop is JIT'd.
- **Greedy / temperature.** At `temperature == 0` verification is exact greedy.
  At `temperature > 0`, speculative sampling is used (accept iff the sampled
  main token equals the draft).
- **Multimodal** (the CLIP `image.projector` on `Q4_K_XL`/`Q5_K_M` tags) is out
  of scope — tinygrad has no vision encoder.

## Correctness

The MTP weights are verified against the GGUF layout by `test/unit/test_mtp.py`
(unit tests build a tiny MTP GGUF and assert module shapes / key popping). The
JIT path is checked for byte-equality against a non-JIT reference at
`temperature=0`, multi-draft runs to full context for K ∈ {1,2,4,8}, multi-turn
reuse matches a cold call, and the fast path stays bit-identical.
