# Running Ollama models with tinygrad

tinygrad can serve the same models you manage with `ollama pull/list/rm`, using
Ollama only as the storage/registry manager and tinygrad as the executor. This
page documents how that works.

## How it works

Ollama keeps an OCI-style content-addressed store. tinygrad reads it directly and
never writes to it:

```
$OLLAMA_MODELS/
├── manifests/<registry>/<namespace>/<name>/<tag>   # JSON manifest (name = dir, tag = file)
└── blobs/sha256-<hex>                              # content-addressed raw blobs
```

`tinygrad.llm.ollama` resolves a model name to the GGUF blob(s) and feeds them to the
existing `Transformer.from_gguf` path — no format conversion, no copying. Two on-disk
formats are handled:

- **Format A (default):** a single `application/vnd.ollama.image.model` layer whose blob
  *is* the whole GGUF file. This is what `ollama pull` produces and is fully supported.
- **Format B (best effort):** a native tensor-per-layer layout
  (`image.config` + `image.tensor; name=; dtype=; shape=` layers with raw bytes). This is
  only produced by `ollama create` from safetensors and is not commonly present. tinygrad
  exposes `resolve_native()` to parse it, but end-to-end serving of format B is **not** wired
  up yet — pass `--native` only if you know what you are doing.

## Pulling and running

```sh
ollama pull qwen3:0.6b
python3 -m tinygrad.llm.cli --model qwen3:0.6b --serve 8000
```

`--model` is resolved in this order:

1. an existing local path or URL;
2. the local Ollama store (a `name[:tag]` or `namespace/name[:tag]`, so a model you
   pulled with `ollama` wins — e.g. `--model qwen3:0.6b --serve`);
3. an exact key in the built-in tinygrad model table (used only when the model is not
   in your Ollama store).

Prefix a name with `ollama://` to force Ollama resolution, e.g. `--model ollama://qwen3:0.6b`.

## Listing and inspecting the store

```bash
python3 -m tinygrad.llm.cli --list-ollama   # every name:tag in the store
python3 -m tinygrad.llm.cli --status          # device + per-model loadability
```

`--status` classifies each model as single-file GGUF, native tensor layers
(`--native`), or cloud/proxied (no local blob).

## Ollama defaults

When the model comes from Ollama, the store's sampling defaults are applied
automatically (from the `application/vnd.ollama.image.params` and `.system` layers):

- `num_ctx` → `--max_context` (clamped to the model's own context length)
- `temperature`, `num_predict`/`max_tokens` → generation defaults
- `stop` → stop strings are honored by the OpenAI-compatible server
- `system` → prepended as the system prompt (server and interactive chat)

Request fields in `/v1/chat/completions` take precedence over these defaults. `top_p`,
`top_k` and `repeat_penalty` are present in the defaults but **not** yet wired into the
sampler; they are ignored with a warning.

## OLLAMA_MODELS

The store root is read from `$OLLAMA_MODELS` (falling back to `~/.ollama/models`).
You can point tinygrad at a different store with `--models-dir`:

```bash
python3 -m tinygrad.llm.cli --model qwen3:0.6b --models-dir /media/smk/Models/Ollama --serve
```

## What tinygrad does NOT run

- GGUF architectures/quants that tinygrad does not implement — loading fails with a clear
  message listing the failing GGML type / arch. Prefer `Q4_K_M`/`Q8_0` builds.
- Format B (native tensor-layer) models end-to-end (best effort only).
- Cloud/proxied Ollama models have no local blob and cannot be served locally.
- Ollama's Go `text/template` chat templates; tinygrad uses the GGUF chat template instead.
