"""Read-only resolver for the local Ollama content-addressed model store.

tinygrad treats Ollama as a purely local model store/registry: Ollama owns the
manifests and blobs, tinygrad only *reads* them. Two GGUF-on-disk formats are
supported:

(A) **single-file GGUF** — one layer of mediaType ``application/vnd.ollama.image.model``
    whose blob is the whole GGUF file (what ``ollama pull`` produces). This is the
    default and is passed untouched to the existing ``Transformer.from_gguf``.
(B) **native tensor-per-layer** — an ``application/vnd.ollama.image.config`` layer plus
    one ``application/vnd.ollama.image.tensor; name=; dtype=; shape=`` layer
    per tensor holding *raw* bytes (not GGUF). Reconstructed in :func:`resolve_native`.

All functions default to ``$OLLAMA_MODELS`` (falling back to ``~/.ollama/models``) but
accept an explicit ``root`` so tests and ``--models-dir`` can point at a fixture store.
"""
from __future__ import annotations
import json, os, pathlib, typing

class OllamaError(RuntimeError): pass
class OllamaNotFound(OllamaError): pass
class ModelNotFoundError(OllamaError): pass
class AmbiguousModelError(OllamaError): pass
class NoModelLayer(OllamaError): pass

MODEL_LAYER = "application/vnd.ollama.image.model"          # (A) whole GGUF blob
TENSOR_LAYER = "application/vnd.ollama.image.tensor"        # (B) raw tensor bytes
CONFIG_LAYER = "application/vnd.ollama.image.config"       # (B) model config
PARAMS_LAYER = "application/vnd.ollama.image.params"       # sampling params (json)
SYSTEM_LAYER = "application/vnd.ollama.image.system"       # system prompt
TOKENIZER_LAYER = "application/vnd.ollama.image.tokenizer"  # (B) tokenizer layer
DOCKER_CONFIG = "application/vnd.docker.container.image.v1+json"  # top-level config obj

# *** low-level store layout ***

def parse_model_name(spec: str) -> tuple[str, str, str]:
  """Split ``[ns/]name[:tag]`` into ``(namespace, name, tag)``.

  ``namespace`` defaults to ``library``, ``tag`` defaults to ``latest``. Registry
  prefixes (``registry.ollama.ai/…``) and an ``ollama://`` scheme are stripped.
  """
  spec = spec.strip().strip("/")
  for prefix in ("ollama://", "registry.ollama.ai/"):
    if spec.startswith(prefix): spec = spec[len(prefix):]
  namespace, sep, rest = spec.partition("/")
  if sep: namespace = namespace or "library"
  else: namespace, rest = "library", spec
  name, sep, tag = rest.rpartition(":")
  if not sep: name, tag = rest, "latest"
  if not name: raise OllamaError(f"invalid model spec {spec!r}")
  return namespace, name, tag

def models_dir(root: pathlib.Path | str | None = None) -> pathlib.Path:
  """Return the Ollama store root: ``root`` > ``$OLLAMA_MODELS`` > ``~/.ollama/models``."""
  if root is None: root = os.environ.get("OLLAMA_MODELS") or str(pathlib.Path.home() / ".ollama" / "models")
  p = pathlib.Path(root)
  if not p.is_dir(): raise OllamaNotFound(f"ollama store not found: {p}")
  return p

def _blob_path(store: pathlib.Path, digest: str) -> pathlib.Path:
  """Map a ``sha256:<hex>`` manifest digest to its blob file path."""
  if not digest.startswith("sha256:"): raise OllamaError(f"invalid digest {digest!r}")
  return store / "blobs" / f"sha256-{digest[len('sha256:'):].lower()}"

def _find_manifest(name: str, root: pathlib.Path | str | None) -> tuple[pathlib.Path, pathlib.Path]:
  """Locate the manifest for ``name``; returns ``(store, manifest_path)``.

  Globs the whole ``manifests/`` tree for manifest files directly under a directory
  named ``<name>`` (the tag is the manifest filename, Ollama's layout). A spec with an
  explicit namespace only matches manifests under that namespace. Zero matches ->
  :class:`ModelNotFoundError`; several -> :class:`AmbiguousModelError`.
  """
  store = models_dir(root)
  namespace, model, tag = parse_model_name(name)
  hits = sorted(p for p in (store / "manifests").glob("**/*") if p.is_file() and p.parent.name == model and p.name == tag)
  # an explicit (non-library) namespace pins the search space
  if namespace != "library": hits = [p for p in hits if p.parent.parent.name == namespace]
  avail = ", ".join(list_models(store)) or "(none)"
  if not hits:
    raise ModelNotFoundError(f"model {name!r} not found in ollama store {store}; available: {avail}")
  if len(hits) > 1:
    raise AmbiguousModelError(f"model {name!r} is ambiguous: {[str(h) for h in hits]}")
  return store, hits[0]

def _read_manifest(store: pathlib.Path, mpath: pathlib.Path) -> dict:
  # read whole file up-front to avoid racing a concurrent `ollama pull` mid-write
  try: raw = mpath.read_bytes()
  except OSError as e: raise OllamaNotFound(f"cannot read manifest {mpath}: {e}") from e
  try: return json.loads(raw)
  except json.JSONDecodeError as e: raise OllamaError(f"malformed manifest {mpath}: {e}") from e

def _layer_blob(store: pathlib.Path, layer: dict, mpath: pathlib.Path) -> pathlib.Path:
  blob = _blob_path(store, layer["digest"])
  if not blob.is_file():
    # a proxied/cloud manifest may reference blobs that were never pulled locally
    raise NoModelLayer(f"blob {layer['digest'][:24]}… for {mpath.name} is not on disk (model not fully pulled?); "
                       f"try `ollama pull` first")
  return blob

# *** public API (A) ***

def list_models(root: pathlib.Path | str | None = None) -> list[str]:
  """Return ``name:tag`` for every manifest in the store (duplicates collapsed)."""
  try: store = models_dir(root)
  except OllamaNotFound: return []
  out = []
  for m in sorted((store / "manifests").glob("*/*/*/*")):
    if m.is_file() and m.parent.is_dir():
      out.append(f"{m.parent.name}:{m.name}")
  # collapse exact duplicates (e.g. library + same user ns)
  return list(dict.fromkeys(out))

def manifest_layers(name: str, root: pathlib.Path | str | None = None
                    ) -> tuple[list[tuple[str, pathlib.Path]], pathlib.Path | None]:
  """Return ``(layers, config_blob_path)`` for ``name``.

  ``layers`` is a list of ``(mediaType, blob_path)`` — there can be several of the same
  mediaType (format-B tensor layers). ``config_blob_path`` is the top-level Docker-style
  config object (``application/vnd.docker.container.image.v1+json``) or ``None``.
  """
  store, mpath = _find_manifest(name, root)
  manifest = _read_manifest(store, mpath)
  layers = [(l.get("mediaType", ""), _layer_blob(store, l, mpath)) for l in (manifest.get("layers") or [])]
  cfg = manifest.get("config", {})
  config_blob = _blob_path(store, cfg["digest"]) if cfg.get("digest") and _blob_path(store, cfg["digest"]).is_file() else None
  return layers, config_blob

def resolve(name: str, root: pathlib.Path | str | None = None) -> str:
  """Return the path of the single-file GGUF blob for an (A) model.

  Raises :class:`NoModelLayer` when the manifest has no ``.model`` layer (it may be a
  native (B) model — see :func:`resolve_native`).
  """
  store, mpath = _find_manifest(name, root)
  manifest = _read_manifest(store, mpath)
  model_layers = [l for l in (manifest.get("layers") or []) if l.get("mediaType", "").startswith(MODEL_LAYER)]
  if len(model_layers) != 1:
    raise NoModelLayer(f"model {name!r} has {len(model_layers)} `application/vnd.ollama.image.model` layers "
                       f"(need exactly 1); this may be a native tensor-per-layer model — try resolve_native")
  return str(_layer_blob(store, model_layers[0], mpath))

# *** public API (C) — sampling/config defaults (Inc 3) ***

def _json_layer(layer_blob: pathlib.Path) -> dict:
  try: return json.loads(layer_blob.read_bytes())
  except (OSError, json.JSONDecodeError): return {}

def ollama_defaults(name: str, root: pathlib.Path | str | None = None) -> dict:
  """Flatten the Ollama config into server-side sampling defaults.

  Reads the top-level config object plus the ``.params`` and ``.system`` layers and
  returns whatever of ``temperature``, ``top_p``, ``top_k``, ``repeat_penalty``
  (float), ``stop`` (list[str]), ``num_predict``/``max_tokens`` (int),
  ``num_ctx`` (int) and ``system`` (str) are present. Empty dict when nothing found.
  """
  layers, config_blob = manifest_layers(name, root)
  out: dict[str, typing.Any] = {}
  # top-level config object: model_format / model_family / file_type
  if config_blob is not None:
    cfg = _json_layer(config_blob) or {}
    for src, dst in (("model_format", "model_format"), ("model_family", "model_family"), ("file_type", "file_type")):
      if cfg.get(src) is not None: out[dst] = cfg[src]
  for media_type, blob in layers:
    mt = media_type.split(";")[0].strip()
    if mt == PARAMS_LAYER:
      params = _json_layer(blob) or {}
      for key in ("temperature", "top_p", "top_k", "repeat_penalty", "stop", "num_predict", "num_ctx", "system"):
        if key in params and params[key] is not None: out[key] = params[key]
      if "num_ctx" in params and out.get("num_ctx") is None: out["num_ctx"] = params["num_ctx"]
      if "num_predict" in params and "max_tokens" not in out: out["max_tokens"] = params["num_predict"]
    elif mt == SYSTEM_LAYER:
      try: sys_txt = blob.read_bytes().decode("utf-8", "replace").strip()
      except OSError: sys_txt = ""
      if sys_txt: out["system"] = sys_txt
  if isinstance(out.get("stop"), str): out["stop"] = [out["stop"]]  # string -> list[str]
  return out

# *** public API (B) — native tensor-per-layer (Inc 4, best effort) ***

def _parse_media_params(media_type: str) -> dict[str, str]:
  out: dict[str, str] = {}
  for part in media_type.split(";")[1:]:
    if "=" in part:
      k, _, v = part.strip().partition("=")
      out[k] = v
  return out

def _dtype_bytes(dtype: str) -> int:
  return {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
          "Q8_0": 1, "Q4_0": 1, "Q4_K": 1, "Q5_K": 1, "Q6_K": 1, "Q4_K_M": 1, "Q5_K_M": 1,
          "Q4_1": 1, "Q5_1": 1, "Q8_K": 1, "F8_E4M3": 1, "F8_E5M2": 1}.get(dtype.upper(), 0)

def resolve_native(name: str, root: pathlib.Path | str | None = None) -> tuple[dict, dict[str, typing.Any]]:
  """Reconstruct ``(kv, raw_tensors)`` for a native tensor-per-layer (B) model.

  ``raw_tensors`` maps tensor name -> a dict carrying the layer blob path, dtype and
  shape, so the caller can hand the raw bytes to ``gguf.ggml_data_to_tensor`` (for GGML
  quant dtypes) or build a ``Tensor`` directly (for native dtypes). ``kv`` is synthesized
  from the config + tokenizer layers. Best effort: unsupported configs raise
  :class:`OllamaError` with a clear message.
  """
  store, mpath = _find_manifest(name, root)
  manifest = _read_manifest(store, mpath)
  cfg_blob = None
  tensors: dict[str, dict[str, typing.Any]] = {}
  kv: dict[str, typing.Any] = {}
  tokenizer_blob = None
  for layer in (manifest.get("layers") or []):
    mt = layer.get("mediaType", "")
    base_mt = mt.split(";")[0].strip()
    blob = _layer_blob(store, layer, mpath)
    if base_mt == CONFIG_LAYER:
      cfg_blob = _json_layer(blob) or {}
    elif base_mt == TENSOR_LAYER:
      params = _parse_media_params(mt)
      n = params.pop("name", "")
      if not n: raise OllamaError(f"format-B tensor layer missing name= in {mt!r}")
      tensors[n] = {"blob": blob, "dtype": params.get("dtype", ""), "shape": params.get("shape", ""), "params": params,
                    "size": layer.get("size")}
    elif base_mt.startswith(TOKENIZER_LAYER):
      tokenizer_blob = blob
  if not tensors:
    raise NoModelLayer(f"model {name!r} has no format-B tensor layers; it may be a single-file (A) model — try resolve()")
  # synthesize kv from config + tokenizer
  if cfg_blob: kv.update({k: str(v) if not isinstance(v, (int, list, dict)) else v for k, v in cfg_blob.items()})
  # tokenizer layer, if present, is a raw GGUF-style or json tokenizer; surface its path
  kv["_tokenizer_blob"] = str(tokenizer_blob) if tokenizer_blob else None
  kv["_tensor_count"] = len(tensors)
  return kv, tensors
