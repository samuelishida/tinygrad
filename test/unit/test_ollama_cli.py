import unittest, tempfile, pathlib, json, hashlib
from tinygrad.llm import ollama as O

class OllamaFixture:
  """Minimal Ollama store tree under a tempdir (mirrors test_ollama.OllamaFixture)."""
  def __init__(self, root: pathlib.Path):
    self.root = root
    (root / "blobs").mkdir(parents=True)
    (root / "manifests" / "registry.ollama.ai").mkdir(parents=True)
  def _digest(self, payload: bytes) -> str:
    h = hashlib.sha256(payload).hexdigest()
    (self.root / "blobs" / f"sha256-{h}").write_bytes(payload)
    return f"sha256:{h}"
  def add_model(self, namespace, name, tag, layers: list[tuple], config_payload: bytes | None = None):
    cfg = {}
    if config_payload is not None:
      cfg = {"mediaType": "application/vnd.docker.container.image.v1+json",
             "digest": self._digest(config_payload), "size": len(config_payload)}
    manifest = {"schemaVersion": 2, "config": cfg,
                "layers": [{"mediaType": t, "digest": self._digest(data), "size": len(data)} for t, data in layers]}
    mdir = self.root / "manifests" / "registry.ollama.ai" / namespace / name
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / tag).write_text(json.dumps(manifest))

GGUF = b"GGUF\x03\x00\x00\x00" + b"\x00" * 16

class TestCliResolve(unittest.TestCase):
  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.root = pathlib.Path(self._tmp.name)
    self.fx = OllamaFixture(self.root)
  def tearDown(self): self._tmp.cleanup()
  def _cli(self):
    from tinygrad.llm import cli
    return cli

  def test_path_passthrough(self):
    self.assertEqual(self._cli()._resolve_model_arg("/tmp/nope.gguf", None, False)[0], "/tmp/nope.gguf")
  def test_mtp_drafts_flag(self):
    p = self._cli()._build_parser()
    self.assertEqual(p.parse_args(["--mtp-drafts", "4"]).mtp_drafts, "4")
    self.assertEqual(p.parse_args(["--mtp-drafts", "auto"]).mtp_drafts, "auto")
    self.assertEqual(p.parse_args([]).mtp_drafts, "1")
  def test_url_passthrough(self):
    url = "https://example.com/m.gguf"
    self.assertEqual(self._cli()._resolve_model_arg(url, None, False)[0], url)
  def test_ollama_store_wins_for_same_name(self):
    # qwen3:0.6b is both a tinygrad dict key AND in the store; the store must win (Inc 2 done criterion)
    self.fx.add_model("library", "qwen3", "0.6b", [(O.MODEL_LAYER, GGUF)], config_payload=b'{"model_format":"gguf"}')
    target, defaults = self._cli()._resolve_model_arg("qwen3:0.6b", str(self.root), False)
    self.assertTrue(pathlib.Path(target).is_file())
    self.assertEqual(defaults.get("model_format"), "gguf")
  def test_ollama_force_prefix(self):
    self.fx.add_model("library", "qwen3", "0.6b", [(O.MODEL_LAYER, GGUF)])
    target, _ = self._cli()._resolve_model_arg("ollama://qwen3:0.6b", str(self.root), False)
    self.assertTrue(pathlib.Path(target).is_file())
  def test_dict_fallback_when_not_in_store(self):
    self._cli().models["zzz-test:1"] = "http://example.test/zzz.gguf"
    try:
      target, defaults = self._cli()._resolve_model_arg("zzz-test:1", str(self.root), False)
      self.assertEqual(target, "http://example.test/zzz.gguf")
      self.assertEqual(defaults, {})
    finally:
      self._cli().models.pop("zzz-test:1", None)
  def test_missing_raises(self):
    with self.assertRaises(SystemExit): self._cli()._resolve_model_arg("no-such-model:1", str(self.root), False)

if __name__ == "__main__": unittest.main()
