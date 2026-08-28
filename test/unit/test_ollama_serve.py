import unittest
from unittest.mock import Mock
from tinygrad.llm.serve import Handler

# fakes for Handler.run_model: a tiny tokenizer mapping id -> decoded text
TOKEN_TEXT = {1: "Hello ", 2: "world ", 3: "two", 4: " more", 5: " done", 6: "! "}

def make_server():
  model = Mock()
  model.get_start_pos = Mock(return_value=0)
  model.generate = Mock(side_effect=lambda toks, temperature=0.0, mtp=False, num_drafts=1, adaptive=False: iter(TOKEN_TEXT.keys()))
  tok = Mock()
  tok.is_end = Mock(side_effect=lambda tid: False)
  return model, tok

def make_decoder():
  def dec(tid=None): return "" if tid is None else TOKEN_TEXT[tid]
  return dec

def collect(stop, model, tok):
  """Run Handler.run_model against fakes and concatenate the content deltas."""
  server = Mock()
  server.model, server.tok, server.default_params = model, tok, {"stop": stop or []}
  server.mtp = False
  server.num_drafts = 1
  server.adaptive = False
  h = Handler.__new__(Handler)
  h.server = server
  tok.stream_decoder = Mock(return_value=make_decoder())
  out = []
  for c in h.run_model([0], "m", stop=stop):
    if (delta := c["choices"][0]["delta"]).get("content"): out.append(delta["content"])
  return "".join(out)

class TestServeStop(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.model, cls.tok = make_server()
  def test_no_stop_full_text(self):
    self.assertEqual(collect(None, self.model, self.tok), "Hello world two more done! ")
  def test_stop_truncates_at_first(self):
    self.assertEqual(collect(["two"], self.model, self.tok), "Hello world ")
  def test_stop_spanning_token(self):
    self.assertEqual(collect(["world"], self.model, self.tok), "Hello ")
  def test_stop_never_matched(self):
    self.assertEqual(collect(["zzz"], self.model, self.tok), "Hello world two more done! ")
  def test_stop_first_token(self):
    self.assertEqual(collect(["Hello "], self.model, self.tok), "")
  def test_stop_multiple_candidates(self):
    # first stop ("more") wins even though "done" also appears later
    self.assertEqual(collect(["more", "done"], self.model, self.tok), "Hello world two ")

if __name__ == "__main__": unittest.main()
