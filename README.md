# nvtokentop

**nvtop for tokens.** A terminal monitor for local LLM servers. It shows live prompt and
generation tokens per second for each model, requests in flight, image renders, and the
memory bandwidth that decode implies, next to CPU, GPU and RAM use.

```
Device 0 [NVIDIA GB10]  GPU 2190MHz  TEMP 70°C  POW 36 W                 nvtokentop 15:13:09  1.0s  q quit +/- interval
CPU[|||                                                 6%] GPU[|||||||||||||||||||||||||||||||||||||||||||||||||| 96%]
MEM IO[||||||||||||||||||||||||||||||    ~166/273 GB/s 61%] RAM[|||||||||||||||||||||||||||||||||||   80.7/121.7Gi 66%]
KV 25%  2 req in flight
     ┌─ generation tok/s ─ ━ gemma-4-12b  ━ qwen3.8-27b ───────────────────────────────────────────────
 100 │
  50 │                                                                                              ┌┐┌
  25 │                                                                                            ──┘└┘
   0 │                                                                                            ─────
     └─────────────────────────────────────────────────────────────────────────────────────────────────
     ┌─ utilization % ─ ━ CPU  ━ GPU  ━ mem IO  ━ RAM  ━ gemma-4-12b KV  ━ qwen3.8-27b KV ─────────────
 100 │                                                                                           ──────
  50 │                                                                                           ─┌────
  25 │                                                                                            │────
   0 │                                                                                           ──────
     └─────────────────────────────────────────────────────────────────────────────────────────────────
    PID MODEL                STATE  SRV      RUN QUE   GEN/s   PRM/s GEN avg PRM avg   ACC  KV% HIT%  TTFT  ITL  GB/s
 367259 google/gemma-4-12b   ready  sglang     0   0     0.0       0     0.0       0     -    0    0     -    -     -
 364333 qwen3.8-27b          ready  sglang     2   0    66.0       0     0.0       0  2.66   25   89     -   29   166
```

It is a single Python file with no dependencies beyond the standard library. It was
built for an NVIDIA DGX Spark (GB10) running SGLang behind
[llama-swap](https://github.com/mostlygeek/llama-swap), and also reads vLLM and llama.cpp.

## Why

Decode on a single-user box is limited by memory bandwidth, not compute. `nvtop` shows GPU
busy, but many GPUs, the GB10 among them, expose no DRAM counter at all. NVML reports 0 %,
and Nsight Systems, Nsight Compute and `perf` find nothing either. Token rates are the number
you actually care about, and the servers already export them.

## Install

```bash
pipx install git+https://github.com/TechBlueprints/nvtokentop
```

Or copy the one file:

```bash
sudo install -m 0755 nvtokentop.py /usr/local/bin/nvtokentop
```

Run it on the GPU host itself. The process table maps each model's port to its process with
`ss` and `nvidia-smi`, so it needs to be local. Python 3.9 or newer is enough. A TOML config
needs 3.11 or newer, and older versions can use a JSON config instead.

## Usage

```bash
nvtokentop                      # TUI: header bars, stacked charts, per-model table
nvtokentop --plain -n 5         # five text samples, no tty needed
nvtokentop --router none --backend vllm=http://127.0.0.1:8001,weights_gb=16 --peak 1008
nvtokentop --list-adapters      # the built-in metric maps, as TOML
```

Keys: `q` quits, `+` and `-` change the sample interval. Charts are drawn in priority order,
as many as the terminal height allows: generation tok/s, utilization %, prompt tok/s, then
requests in flight.

## Where the numbers come from

Everything is read-only.

| Column | Source |
|---|---|
| GEN/s, PRM/s | Per-step token counters on each server, so they move while a request runs |
| GEN avg, PRM avg | Completion counters averaged over `--window` seconds |
| RUN, QUE | llama-swap's event stream when present, else the server's running and queued gauges |
| ACC | Speculative-decoding accept length |
| KV%, HIT% | KV cache use and prefix-cache hit rate |
| TTFT, ITL | Time to first token and inter-token latency, from the server's histograms |
| GB/s | Derived memory bandwidth, explained below |
| CPU, RAM | `/proc/stat` and `/proc/meminfo` |
| GPU | `nvidia-smi` |

**GB/s is derived, not measured.** Every decode forward pass streams the resident weights once
and reads the KV cache of the sequences being decoded. So:

```
GB/s ≈ (weights + KV of running sequences) × forward passes per second
```

Passes per second come from a counter that ticks once per decode pass. The percentage is
against the GPU's bandwidth spec. It ignores activations and cache hits, so treat it as an
estimate. On a GB10 it reads 190 to 260 GB/s of the 273 GB/s spec during single-stream decode,
which is what you expect at the memory roofline.

**Image models.** A diffusion server with no `/metrics` endpoint is still shown. With
llama-swap, its row lists images in flight, images done and seconds per image, taken from
llama-swap's request events. SGLang diffusion servers are recognized by `/model_info`.

## Servers

| Adapter | Live token rate | Passes | Weight size |
|---|---|---|---|
| `sglang` | `realtime_tokens_total` | `cuda_graph_passes_total{mode="decode_cuda_graph"}` | `weight_memory_usage_gb` |
| `vllm` (v1 engine) | `prompt_tokens_total`, `generation_tokens_total` | `iteration_tokens_total_count` | set `weights_gb` |
| `llamacpp` (`llama-server --metrics`) | growth of `n_decode_total × n_busy_slots_per_decode` | `n_decode_total` | set `weights_gb` |

The adapter is picked from the metric prefix, and the SRV column shows which one was used.
The SGLang adapter was checked against a live server. The vLLM and llama.cpp adapters follow
the metric names in those projects' source as of November 2026, and so far have only been
tested against recorded output. Reports from real servers are welcome.

llama.cpp counts generated tokens only when a request finishes, so its live rate is
estimated from decode calls and busy slots, which also counts some prompt work.

## Configuration

nvtokentop reads the first file it finds: `--config`, `$NVTOKENTOP_CONFIG`,
`~/.config/nvtokentop/config.toml`, then `/etc/nvtokentop.toml`. With no file it watches
llama-swap on `127.0.0.1:8000`. See [examples/config.toml](examples/config.toml).

- **Model source.** `router = "llama-swap"` or `"none"`, plus any number of `[[backends]]`
  with a `name` and `url`, or `--backend NAME=URL[,adapter=X,weights_gb=N]` on the command
  line.
- **Adapters.** `[adapters.<name>]` overrides keys of a built-in adapter, or defines a new one
  with `base = "<built-in>"` and its own `prefix`. Run `--list-adapters` to see every key.
- **Weights.** `weights_gb` on a backend, or under `[models."<name>"]` for a model llama-swap
  lists. Without it, GB/s shows `-` for servers that do not export their weight size.
- **Bandwidth peak.** `peak_gbps` or `--peak`. Otherwise it is looked up from the GPU name
  in a built-in table of spec figures. An unknown GPU shows GB/s with no percentage.

## Limitations

- Only the first GPU is shown in the header.
- The RAM bar is system memory. On unified-memory chips such as the GB10 that is the real
  memory pressure. On discrete GPUs, VRAM is not shown yet.
- No Windows or macOS support: it reads `/proc`.

## Credits

The layout is inspired by [nvtop](https://github.com/Syllo/nvtop). nvtokentop shares no code
with it and is not affiliated with that project.

## License

Apache License 2.0. See [LICENSE](LICENSE).
