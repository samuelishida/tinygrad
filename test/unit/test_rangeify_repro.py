"""Minimal reproducers for the rangeify crashes on symbolic-extent graphs.

Two patterns (see .plans/rangeify-symbolic/repro.md in the FreeToken repo):

(a) REDUCE has no ranges in rangeify — the MTP variable-draft verify capture
    (symbolic `toks` 1..max_drafts+1) runs the model blocks through the
    rangeify scheduler, which raises in convert_reduce_to_reduce_with_ranges
    (tinygrad/schedule/indexing.py:115) for a REDUCE not registered in the
    range map.

(b) IndexError in run_rangeify — the real-model prefill capture at
    max_context=16384: a scalar CONST consumed by a MAX (the
    `maximum(1e-8)` in kv_q8_quantize_batched) whose shape (2,1,2,n_toks,8,1)
    has a trailing 1-dim that the consumer chain (division -> reshape ->
    bitcast) merges away, so the MAX's inherited ranges (5) are shorter than
    its shape (6) and `broadcast_axes((), c.shape)` indexes out of bounds.
    Reproduced by `scripts/bench-tinygrad.py --model ... --ctx 16384`
    (the engine build crashes in the prefill capture); the OOB context is
    captured in repro.md.

Both must FAIL before the rangeify fix and PASS after.
"""
import unittest

from tinygrad import Tensor, UOp
from tinygrad.llm.model import Transformer, TransformerConfig, TransformerBlock, MTPBlock
from tinygrad.nn.state import get_state_dict, load_state_dict

DIM, HIDDEN, N_HEADS, VOCAB, CTX = 16, 32, 2, 32, 32


def _config(num_blocks=4, max_context=CTX):
  return TransformerConfig(num_blocks=num_blocks, dim=DIM, hidden_dim=HIDDEN, n_heads=N_HEADS,
    n_kv_heads=N_HEADS, norm_eps=1e-5, vocab_size=VOCAB, head_dim=DIM//N_HEADS, rope_theta=10000,
    rope_dim=DIM//N_HEADS, v_head_dim=DIM//N_HEADS, max_context=max_context)


def _make_model(num_blocks=4, max_context=CTX):
  c = _config(num_blocks, max_context)
  m = Transformer(c)
  m.mtp = MTPBlock(c, TransformerBlock)
  sd = get_state_dict(m)
  for k in sd: sd[k] = Tensor.uniform(*sd[k].shape, low=-0.1, high=0.1)
  load_state_dict(m, sd, verbose=False)
  return m


class TestRangeifyRepro(unittest.TestCase):
  def test_reduce_no_ranges(self):
    """(a) the MTP generate (rollout) capture must not crash the rangeify."""
    m = _make_model()
    out = list(m.generate([1, 2, 3, 4], temperature=0.0, mtp=True))
    self.assertTrue(all(isinstance(t, int) for t in out))
    self.assertGreater(len(out), 0)


if __name__ == "__main__":
  unittest.main()
