import unittest, tempfile, pathlib, json, hashlib, os
from tinygrad.llm import ollama as O

class OllamaFixture:
  """Build a minimal Ollama store tree under a tempdir and point OLLAMA_MODELS at it."""
  def __init__(self, root: pathlib.Path):
    self.root = root
    (root / "blobs").mkdir(parents=True)
    (root / "manifests" / "registry.ollama.ai").mkdir(parents=True)
    self._digests = {}
  def _digest(self, payload: bytes) -> str:
    h = hashlib.sha256(payload).hexdigest()
    (self.root / "blobs" / f"sha256-{h}").write_bytes(payload)
    return f"sha256:{h}"
  def add_model(self, namespace, name, tag, layers: list[dict], config_payload: bytes | None = None) -> pathlib.Path:
    cfg = {}
    if config_payload is not None:
      cfg = {"mediaType": "application/vnd.docker.container.image.v1+json",
             "digest": self._digest(config_payload), "size": len(config_payload)}
    manifest = {"schemaVersion": 2, "config": cfg,
                "layers": [{"mediaType": t, "digest": self._digest(data), "size": len(data)} for t, data in layers]}
    mdir = self.root / "manifests" / "registry.ollama.ai" / namespace / name
    mdir.mkdir(parents=True, exist_ok=True)
    mpath = mdir / tag
    mpath.write_text(json.dumps(manifest))
    return mpath
  def add_params(self, **kw) -> bytes:
    return json.dumps(kw).encode()

class TestOllamaResolver(unittest.TestCase):
  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.root = pathlib.Path(self._tmp.name)
    self.fx = OllamaFixture(self.root)
  def tearDown(self): self._tmp.cleanup()
  def _r(self, *a): return O.resolve(*a, root=self.root)
  def _lm(self, *a): return O.list_models(*a) if a else O.list_models(root=self.root)

  GGUF = b"GGUF\x03\x00\x00\x00" + b"\x00" * 16

  # *** parse_model_name ***
  def test_parse_bare(self): self.assertEqual(O.parse_model_name("qwen3"), ("library", "qwen3", "latest"))
  def test_parse_tag(self): self.assertEqual(O.parse_model_name("qwen3:0.6b"), ("library", "qwen3", "0.6b"))
  def test_parse_ns_tag(self): self.assertEqual(O.parse_model_name("smtek/Qwen3.8-27B:Q4_K_XL"), ("smtek", "Qwen3.8-27B", "Q4_K_XL"))
  def test_parse_ns_only(self): self.assertEqual(O.parse_model_name("smtek/Qwen3.8-27B"), ("smtek", "Qwen3.8-27B", "latest"))
  def test_parse_ollama_scheme(self): self.assertEqual(O.parse_model_name("ollama://qwen3:0.6b"), ("library", "qwen3", "0.6b"))
  def test_parse_registry_prefix(self): self.assertEqual(O.parse_model_name("registry.ollama.ai/library/qwen3:0.6b"), ("library", "qwen3", "0.6b"))
  def test_parse_invalid(self):
    with self.assertRaises(O.OllamaError): O.parse_model_name(":0.6b")
    with self.assertRaises(O.OllamaError): O.parse_model_name("ns/:tag")

  # *** models_dir ***
  def test_models_dir_explicit(self): self.assertEqual(O.models_dir(self.root), self.root)
  def test_models_dir_missing(self):
    with self.assertRaises(O.OllamaNotFound): O.models_dir(self.root / "nope" / "nope")
  def test_models_dir_env(self):
    old = os.environ.get("OLLAMA_MODELS")
    os.environ["OLLAMA_MODELS"] = str(self.root)
    try: self.assertEqual(O.models_dir(), self.root)
    finally:
      if old is None: os.environ.pop("OLLAMA_MODELS", None)
      else: os.environ["OLLAMA_MODELS"] = old

  # *** list_models ***
  def test_list_empty(self): self.assertEqual(self._lm(), [])
  def test_list(self):
    self.fx.add_model("library", "qwen3", "0.6b", [(O.MODEL_LAYER, self.GGUF)])
    self.fx.add_model("library", "qwen3", "1.7b", [(O.MODEL_LAYER, b"GGUF\x03" + b"\x01")])
    self.fx.add_model("smtek", "MyModel", "latest", [(O.MODEL_LAYER, b"GGUF\x03" + b"\x02")])
    models = self._lm()
    self.assertIn("qwen3:0.6b", models)
    self.assertIn("qwen3:1.7b", models)
    self.assertIn("MyModel:latest", models)

  # *** resolve (format A) ***
  def test_resolve_returns_model_blob(self):
    self.fx.add_model("library", "qwen3", "0.6b",
                      [(O.MODEL_LAYER, self.GGUF), (O.PARAMS_LAYER, b'{"temperature":0.6}')],
                      config_payload=b'{"model_format":"gguf"}')
    resolved = pathlib.Path(self._r("qwen3:0.6b"))
    self.assertTrue(resolved.is_file())
    self.assertEqual(resolved.read_bytes(), self.GGUF)
    self.assertTrue(str(resolved).startswith(str(self.root / "blobs" / "sha256-")))
  def test_resolve_latest_default(self):
    self.fx.add_model("library", "qwen3", "latest", [(O.MODEL_LAYER, self.GGUF)])
    self.assertTrue(pathlib.Path(self._r("qwen3")).is_file())
  def test_resolve_two_model_layers_rejected(self):
    self.fx.add_model("library", "qwen3", "0.6b",
                      [(O.MODEL_LAYER, self.GGUF), (O.MODEL_LAYER, b"OTHER" + b"\x00" * 20)])
    with self.assertRaises(O.NoModelLayer): self._r("qwen3:0.6b")
  def test_resolve_missing_model_layer(self):
    self.fx.add_model("library", "cloudy", "cloud",
                      [(O.TENSOR_LAYER + "; name=tt; dtype=F32; shape=4", b"\x00" * 16)])
    with self.assertRaises(O.NoModelLayer): self._r("cloudy:cloud")
  def test_resolve_missing_blob(self):
    # manifest references a digest whose blob file was never written
    h = hashlib.sha256(b"missing").hexdigest()
    mdir = self.root / "manifests" / "registry.ollama.ai" / "library" / "ghost"
    mdir.mkdir(parents=True)
    (mdir / "latest").write_text(json.dumps({"schemaVersion": 2, "layers": [
      {"mediaType": O.MODEL_LAYER, "digest": f"sha256:{h}", "size": 7}]}))
    with self.assertRaises(O.NoModelLayer): self._r("ghost:latest")
  def test_resolve_missing_model(self):
    with self.assertRaises(O.ModelNotFoundError): self._r("nope:1")
  def test_path_traversal_rejected(self):
    with self.assertRaises(O.ModelNotFoundError): self._r("../../etc/passwd")

  # *** manifest_layers ***
  def test_manifest_layers_shares(self):
    self.fx.add_model("library", "qwen3", "0.6b",
                      [(O.MODEL_LAYER, self.GGUF), ("application/vnd.ollama.image.template", b'{{.}}')],
                      config_payload=b'{"file_type":"Q4_K_M"}')
    layers, cfg = O.manifest_layers("qwen3:0.6b", root=self.root)
    by_mt = {}
    for mt, p in layers: by_mt.setdefault(mt, []).append(p)
    self.assertEqual(by_mt[O.MODEL_LAYER][0].read_bytes(), self.GGUF)
    self.assertIsNotNone(cfg)
    self.assertEqual(json.loads(cfg.read_bytes())["file_type"], "Q4_K_M")
  def test_manifest_layers_config_none(self):
    self.fx.add_model("library", "qwen3", "0.6b", [(O.MODEL_LAYER, self.GGUF)])
    _, cfg = O.manifest_layers("qwen3:0.6b", root=self.root)
    self.assertIsNone(cfg)

  # *** ambiguity ***
  def test_ambiguous_rejected(self):
    self.fx.add_model("library", "dup", "t", [(O.MODEL_LAYER, self.GGUF)])
    self.fx.add_model("user", "dup", "t", [(O.MODEL_LAYER, self.GGUF)])
    self.fx.add_model("other", "dup", "t", [(O.MODEL_LAYER, self.GGUF)])
    self.fx.add_model("library", "dup", "t", [(O.MODEL_LAYER, b"Z" + b"\x00" * 20)])  # same ns/tag, different file name? no—same path dedupe
    with self.assertRaises(O.AmbiguousModelError): self._r("dup:t")  # same name:tag across 3 namespaces, no ns pin
  def test_namespace_pins_disambiguation(self):
    self.fx.add_model("library", "dup", "t", [(O.MODEL_LAYER, self.GGUF)])
    self.fx.add_model("user", "dup", "t", [(O.MODEL_LAYER, self.GGUF)])
    resolved = pathlib.Path(self._r("user/dup:t"))
    self.assertTrue(resolved.is_file())

  # *** ollama_defaults ***
  def test_defaults_params_and_config(self):
    self.fx.add_model("library", "qwen3", "0.6b",
                      [(O.MODEL_LAYER, self.GGUF),
                       (O.PARAMS_LAYER, self.fx.add_params(temperature=0.6, top_p=0.95, top_k=20,
                                                            repeat_penalty=1, stop=["\n\n"], num_predict=32, num_ctx=8192))],
                      config_payload=b'{"model_format":"gguf","model_family":"qwen3","file_type":"Q4_K_M"}')
    d = O.ollama_defaults("qwen3:0.6b", root=self.root)
    self.assertEqual(d["temperature"], 0.6)
    self.assertEqual(d["top_p"], 0.95)
    self.assertEqual(d["top_k"], 20)
    self.assertEqual(d["repeat_penalty"], 1)
    self.assertEqual(d["stop"], ["\n\n"])
    self.assertEqual(d["num_ctx"], 8192)
    self.assertEqual(d["max_tokens"], 32)  # from num_predict
    self.assertEqual(d["model_format"], "gguf")
    self.assertEqual(d["file_type"], "Q4_K_M")
  def test_defaults_empty_when_absent(self):
    self.fx.add_model("library", "bare", "t", [(O.MODEL_LAYER, self.GGUF)])
    self.assertEqual(O.ollama_defaults("bare:t", root=self.root), {})
  def test_defaults_system_layer(self):
    self.fx.add_model("library", "sys", "t", [(O.MODEL_LAYER, self.GGUF), (O.SYSTEM_LAYER, b"You are helpful.\n")])
    self.assertEqual(O.ollama_defaults("sys:t", root=self.root)["system"], "You are helpful.")
  def test_defaults_stop_string_normalized(self):
    self.fx.add_model("library", "s", "t", [(O.MODEL_LAYER, self.GGUF),
                                            (O.PARAMS_LAYER, b'{"stop":"STOP"}')])
    self.assertEqual(O.ollama_defaults("s:t", root=self.root)["stop"], ["STOP"])

  # *** resolve_native (format B) ***
  def test_native_reconstructs(self):
    self.fx.add_model("library", "native", "v1",
                      [(O.CONFIG_LAYER + "; type=gguf", json.dumps({"architectures": ["qwen2"], "dim": 4,
                                                                     "context_length": 1024}).encode()),
                       (O.TENSOR_LAYER + "; name=embd.weight; dtype=F32; shape=10 4", b"\x00" * 16),
                       (O.TENSOR_LAYER + "; name=blk.0.attn.wq.weight; dtype=Q4_K_M; shape=2 2", b"\x00" * 8)],
                      config_payload=b'{"model_format":"native"}')
    kv, tensors = O.resolve_native("native:v1", root=self.root)
    self.assertEqual(kv["architectures"], ["qwen2"])
    self.assertEqual(len(tensors), 2)
    self.assertEqual(tensors["embd.weight"]["dtype"], "F32")
    self.assertTrue(tensors["blk.0.attn.wq.weight"]["blob"].is_file())
  def test_native_missing_rejected(self):
    self.fx.add_model("library", "plain", "t", [(O.MODEL_LAYER, self.GGUF)])
    with self.assertRaises(O.NoModelLayer): O.resolve_native("plain:t", root=self.root)

  # *** against the real store (skip if absent) ***
  @unittest.skipUnless(os.path.isdir(os.environ.get("OLLAMA_MODELS", "")) and
                       os.path.exists(str(pathlib.Path(os.environ["OLLAMA_MODELS"]) / "manifests")),
                       "requires a real OLLAMA_MODELS store")
  def test_real_store_resolve(self):
    p = O.resolve("qwen3:0.6b")
    self.assertTrue(os.path.isfile(p))
    with open(p, "rb") as f: self.assertEqual(f.read(4), b"GGUF")

if __name__ == "__main__": unittest.main()
