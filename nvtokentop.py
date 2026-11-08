#!/usr/bin/env python3
"""nvtokentop -- nvtop for tokens.

Copyright 2026 Clint Goudie-Nice.  Licensed under the Apache License, Version 2.0.
Inspired by nvtop (https://github.com/Syllo/nvtop); shares no code with it.

An nvtop-style view of local model servers: live prompt and generation token rates
per LLM, images in flight and seconds per image for a diffusion model, requests in
flight per model, and the memory bandwidth decode implies.  Many GPUs (the GB10 among
them) expose no DRAM counter (NVML, nsys, ncu, perf all come up empty), so bandwidth
is derived: every decode forward pass streams the resident weights once and reads the
KV cache of the running sequences, so bytes-per-pass x passes-per-second is the
effective figure, compared with the GPU's memory bandwidth spec.

Pluggable parts (all optional; with no config it watches llama-swap on 127.0.0.1:8000):
  * Model source: llama-swap (/running plus its /api/events stream) or none, plus any
    number of static backends (config [[backends]] or --backend NAME=URL).
  * Metrics adapter per backend: built-in sglang, vllm and llamacpp specs map each
    server's Prometheus names onto the columns; picked by metric prefix, forced per
    backend, overridden or extended in the config ([adapters.<name>], "base" to
    inherit).  --list-adapters prints the built-ins as TOML to start from.
  * Bandwidth peak: peak_gbps in the config or --peak, else looked up from the GPU
    name (PEAKS below), else the MEM IO bar shows GB/s without a percentage.
  * Weights for servers that do not export them (vLLM, llama.cpp): weights_gb per
    backend or per model ([models."<name>"]).
Config: --config, $NVTOKENTOP_CONFIG, ~/.config/nvtokentop/config.toml, then
/etc/nvtokentop.toml (TOML, or JSON when the name ends in .json).

Data sources (all read-only; SGLang measured on a DGX Spark 2026-11-02 so the columns
mean what they say, vLLM and llama.cpp names taken from their source 2026-11-08):
  * SGLang /metrics on each LLM backend.  realtime_tokens_total{mode=decode|
    prefill_compute|prefill_cache} and cuda_graph_passes_total{mode=decode_cuda_graph}
    tick per scheduler step, so the live columns, the charts and ~GB/s update at the
    sampling interval.  prompt_tokens_total / generation_tokens_total only move when a
    request finishes (one lump) and feed the --window averages; gen_throughput,
    spec_accept_length, token_usage, decode_sum_seq_lens are gauges refreshed every
    --decode-log-interval steps (40 by default); TTFT / ITL come from the histograms.
  * vLLM (v1 engine) /metrics: prompt_tokens_total and generation_tokens_total are
    bumped every engine step, so they are live; iteration_tokens_total_count counts
    steps (passes).  No weight-size metric: set weights_gb for GB/s.
  * llama.cpp server --metrics: tokens_predicted_total only moves when a request
    finishes, so live generation is the growth of n_decode_total x
    n_busy_slots_per_decode (busy slots summed over decode calls); n_decode_total
    counts passes.  No weight-size or latency histograms.
  * llama-swap /api/events (SSE): one "inflight" record per request with model, path,
    elapsed ms, resp bytes; upsert on start/progress, remove on finish.  This is the
    only source for the diffusion model (its server has no /metrics) and gives live
    requests-in-flight for every model.
  * llama-swap /running for the model list, nvidia-smi for the GPU line and per
    process memory, ss + /proc to map a model's port to its scheduler process,
    /proc/stat for CPU busy % and /proc/meminfo for RAM used (the GB10's one memory
    pool; NVML reports GPU memory as N/A).

Runs on the GPU host itself (the process table needs local ss and nvidia-smi):

    nvtokentop                 # TUI: header bars, stacked charts, per-model table
    nvtokentop --plain         # one block per interval (no tty needed)
    nvtokentop --plain -n 5    # five samples and exit
    nvtokentop --router none --backend big=http://127.0.0.1:8001 --peak 1008
    nvtokentop --list-adapters # built-in metric maps as TOML
    ssh -t gpu-host nvtokentop # from another machine

Keys: q quit, +/- change the interval.
"""

import argparse
import curses
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime

__version__ = "0.1.0"

try:
    import tomllib
except ImportError:  # Python < 3.11: JSON configs only
    tomllib = None

# Memory bandwidth specs (GB/s), matched as a substring of nvidia-smi's GPU name, first
# match wins (so "4060 Ti" sits before "4060").  Override with peak_gbps / --peak.
PEAKS = [
    ("GB10", 273.0), ("Thor", 273.0), ("AGX Orin", 204.8), ("Orin NX", 102.4),
    ("RTX PRO 6000", 1792.0), ("RTX 5090", 1792.0), ("RTX 5080", 960.0), ("RTX 5070 Ti", 896.0),
    ("RTX 5070", 672.0), ("RTX 5060 Ti", 448.0), ("RTX 4090", 1008.0), ("RTX 4080 SUPER", 736.0),
    ("RTX 4080", 716.8), ("RTX 4070 Ti SUPER", 672.0), ("RTX 4070", 504.0),
    ("RTX 4060 Ti", 288.0), ("RTX 4060", 272.0), ("RTX 3090 Ti", 1008.0), ("RTX 3090", 936.0),
    ("RTX 3080", 760.0), ("RTX 6000 Ada", 960.0), ("L40S", 864.0), ("L40", 864.0), ("L4", 300.0),
    ("H200", 4800.0), ("H100 NVL", 3900.0), ("H100 PCIe", 2000.0), ("H100", 3350.0),
    ("A100-SXM4-80GB", 2039.0), ("A100 80GB", 1935.0), ("A100", 1555.0),
    ("Quadro P1000", 82.0),
]

# Metric maps.  A string is one metric; a list is several: counters in a list are
# summed, gauges take the first one present.  "name/<mode>" means that metric with
# mode= or stage= set to <mode> (see parse_prom).
#   prompt_total, gen_total, requests_total  completion counters (avg columns, totals)
#   gen_live / prompt_live / prompt_cached_live  per-step counters for the live columns
#   gen_live_product [gauge, counter]  live gen = growth of gauge x counter per second
#   gen_gauge, prompt_fallback          used when the per-step counters are absent
#   running, queued, kv_usage (0..1), cache_hit (0..1), accept_len  gauges
#   cache_hit_counters [hits, queries], accept_counters [accepted, drafts]
#   weights (+ weights_unit GiB|GB), kv_total (+ kv_unit), kv_max_tokens, kv_seq_tokens
#   passes  counter that ticks once per decode forward pass
#   ttft, itl  histogram base names (_sum / _count are read)
ADAPTERS = {
    "sglang": {
        "prefix": "sglang:",
        "prompt_total": "sglang:prompt_tokens_total",
        "gen_total": "sglang:generation_tokens_total",
        "requests_total": "sglang:num_requests_total",
        "gen_live": ["sglang:realtime_tokens_total/decode"],
        "prompt_live": ["sglang:realtime_tokens_total/prefill_compute",
                        "sglang:realtime_tokens_total/prefill_cache"],
        "prompt_cached_live": ["sglang:realtime_tokens_total/prefill_cache"],
        "live_label": "step counters",
        "gen_gauge": "sglang:gen_throughput",
        "prompt_fallback": ["sglang:prefill_effective_tokens_total"],
        "fallback_label": "40-step gauges",
        "running": "sglang:num_running_reqs",
        "queued": "sglang:num_queue_reqs",
        "kv_usage": "sglang:token_usage",
        "cache_hit": "sglang:cache_hit_rate",
        "accept_len": "sglang:spec_accept_length",
        "weights": "sglang:weight_memory_usage_gb",
        "weights_unit": "GiB",
        "kv_total": "sglang:kv_cache_memory_usage_gb",
        "kv_unit": "GiB",
        "kv_max_tokens": "sglang:max_total_num_tokens",
        "kv_seq_tokens": "sglang:decode_sum_seq_lens",
        "passes": ["sglang:cuda_graph_passes_total/decode_cuda_graph"],
        "ttft": "sglang:time_to_first_token_seconds",
        "itl": "sglang:inter_token_latency_seconds",
    },
    "vllm": {
        "prefix": "vllm:",
        "prompt_total": "vllm:prompt_tokens_total",
        "gen_total": "vllm:generation_tokens_total",
        "requests_total": "vllm:request_success_total",
        "gen_live": ["vllm:generation_tokens_total"],
        "prompt_live": ["vllm:prompt_tokens_total"],
        "prompt_cached_live": ["vllm:prompt_tokens_cached_total"],
        "live_label": "per-step counters",
        "running": "vllm:num_requests_running",
        "queued": "vllm:num_requests_waiting",
        "kv_usage": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
        "cache_hit_counters": ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"],
        "accept_counters": ["vllm:spec_decode_num_accepted_tokens_total",
                            "vllm:spec_decode_num_drafts_total"],
        "passes": ["vllm:iteration_tokens_total_count"],
        "ttft": "vllm:time_to_first_token_seconds",
        "itl": ["vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"],
    },
    "llamacpp": {
        "prefix": "llamacpp:",
        "prompt_total": "llamacpp:prompt_tokens_total",
        "gen_total": "llamacpp:tokens_predicted_total",
        "gen_live_product": ["llamacpp:n_busy_slots_per_decode", "llamacpp:n_decode_total"],
        "prompt_live": ["llamacpp:prompt_tokens_total", "llamacpp:prompt_tokens_cached_total"],
        "prompt_cached_live": ["llamacpp:prompt_tokens_cached_total"],
        "live_label": "decode calls x busy slots",
        "running": "llamacpp:requests_processing",
        "queued": "llamacpp:requests_deferred",
        "kv_usage": "llamacpp:kv_cache_usage_ratio",
        "accept_counters": ["llamacpp:spec_decode_num_accepted_tokens_total",
                            "llamacpp:spec_decode_num_drafts_total"],
        "passes": ["llamacpp:n_decode_total"],
    },
}

GIB = 1024 ** 3
GB = 1e9
HIST_LEN = 600
SAMPLE_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)")
LABEL_RE = re.compile(r'(?:mode|stage)="([^"]+)"')


def http_get(url, timeout=2.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (local only)
        return resp.read().decode("utf-8", "replace")


def parse_prom(text):
    """name -> value summed over label sets, plus name/<mode> for samples that carry a
    mode= or stage= label.  Histogram buckets are dropped; _sum and _count are kept."""
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = SAMPLE_RE.match(line)
        if not m:
            continue
        name, labels, val = m.groups()
        if name.endswith("_bucket"):
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + v
        if labels:
            lm = LABEL_RE.search(labels)
            if lm:
                key = f"{name}/{lm.group(1)}"
                out[key] = out.get(key, 0.0) + v
    return out


def fmt_int(n):
    n = int(n)
    if n >= 10_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 100_000:
        return f"{n / 1e3:.0f}k"
    return f"{n:,}"


def ms(v):
    return "-" if v is None else f"{v:.0f}"


# ----------------------------------------------------------------------------- config and adapters

def as_list(v):
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def gauge(m, keys):
    """First of keys present in m, else None."""
    for k in as_list(keys):
        if k in m:
            return m[k]
    return None


def counter(m, keys):
    """Sum of the keys present in m, or None when none are."""
    vals = [m[k] for k in as_list(keys) if k in m]
    return sum(vals) if vals else None


def config_paths(explicit):
    if explicit:
        return [explicit]
    paths = []
    if os.environ.get("NVTOKENTOP_CONFIG"):
        paths.append(os.environ["NVTOKENTOP_CONFIG"])
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    paths += [os.path.join(xdg, "nvtokentop", "config.toml"), "/etc/nvtokentop.toml"]
    return paths


def load_config(explicit):
    """(config dict, path or None).  An explicit path that is missing is an error."""
    for path in config_paths(explicit):
        if not os.path.exists(path):
            if path == explicit:
                sys.exit(f"nvtokentop: config {path} not found")
            continue
        with open(path, "rb") as f:
            raw = f.read()
        if path.endswith(".json"):
            return json.loads(raw), path
        if tomllib is None:
            sys.exit(f"nvtokentop: {path} is TOML but this Python has no tomllib (3.11+); use .json")
        return tomllib.loads(raw.decode("utf-8")), path
    return {}, None


def load_adapters(cfg):
    """Built-in adapters merged with [adapters.<name>] from the config.  A config entry
    with the name of a built-in overrides its keys; a new name starts from "base"."""
    adapters = {k: dict(v) for k, v in ADAPTERS.items()}
    for name, spec in (cfg.get("adapters") or {}).items():
        spec = dict(spec)
        base = spec.pop("base", None)
        merged = dict(adapters.get(base or name, {}))
        merged.update(spec)
        if not merged.get("prefix"):
            sys.exit(f"nvtokentop: adapter {name} needs a prefix (or a base to inherit one)")
        adapters[name] = merged
    return adapters


def pick_adapter(adapters, metrics, forced=None):
    if forced:
        if forced not in adapters:
            raise KeyError(forced)
        return forced
    for name, spec in adapters.items():
        p = spec.get("prefix", "")
        if p and any(k.startswith(p) for k in metrics):
            return name
    return None


def toml_value(v):
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(str(v))


def print_adapters(adapters):
    print("# nvtokentop metric adapters.  Copy one into your config under [adapters.<name>],")
    print("# keep the keys you change, or add base = \"<built-in>\" to a new name.")
    for name, spec in adapters.items():
        print(f"\n[adapters.{name}]")
        for k, v in spec.items():
            print(f"{k} = {toml_value(v)}")


def peak_for(name):
    for needle, gbps in PEAKS:
        if needle.lower() in (name or "").lower():
            return gbps, needle
    return 0.0, None


# ----------------------------------------------------------------------------- llama-swap events

class Inflight(threading.Thread):
    """Follows llama-swap's /api/events stream and keeps, per model, the requests in
    flight and a history of completed requests (finish time, elapsed ms, path)."""

    available = True

    def __init__(self, llama_swap):
        super().__init__(daemon=True)
        self.url = llama_swap.rstrip("/") + "/api/events"
        self.lock = threading.Lock()
        self.live = {}        # request id -> record
        self.done = deque(maxlen=5000)  # (t_finish, model, elapsed_ms, req_path)
        self.counts = {}      # model -> completed this session
        self.connected = False
        self.error = None

    def run(self):
        while True:
            try:
                req = urllib.request.Request(self.url, headers={"Accept": "text/event-stream"})
                with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                    self.connected, self.error = True, None
                    for raw in resp:
                        line = raw.decode("utf-8", "replace").rstrip("\r\n")
                        if line.startswith("data:"):
                            self._handle(line[5:].strip())
            except (urllib.error.URLError, OSError, ValueError) as e:
                self.connected, self.error = False, str(getattr(e, "reason", e))[:60]
            time.sleep(2)

    def _handle(self, payload):
        try:
            ev = json.loads(payload)
            if ev.get("type") != "inflight":
                return
            d = json.loads(ev.get("data", "{}"))
        except (ValueError, TypeError):
            return
        op = d.get("operation")
        now = time.time()
        with self.lock:
            if op == "snapshot":
                self.live = {r["id"]: r for r in d.get("requests", []) if "id" in r}
            elif op == "upsert":
                r = d.get("request") or {}
                if "id" in r:
                    self.live[r["id"]] = r
            elif op == "remove":
                r = self.live.pop(d.get("id"), None)
                if r:
                    model = r.get("model", "?")
                    self.done.append((now, model, r.get("elapsed_ms", 0) or 0, r.get("req_path", "")))
                    self.counts[model] = self.counts.get(model, 0) + 1

    def stats(self, model, window, now):
        """inflight, oldest in-flight age s, completed in window, avg elapsed ms in
        window, last elapsed ms, completed this session."""
        with self.lock:
            live = [r for r in self.live.values() if r.get("model") == model]
            recent = [(t, e) for (t, m, e, _p) in self.done if m == model and now - t <= window]
            last = next((e for (t, m, e, _p) in reversed(self.done) if m == model), None)
            total = self.counts.get(model, 0)
        age = 0.0
        for r in live:
            try:
                ts = datetime.fromisoformat(r.get("timestamp", "").replace("Z", "+00:00")).timestamp()
                age = max(age, now - ts)
            except ValueError:
                pass
        avg = sum(e for _t, e in recent) / len(recent) if recent else None
        return {"inflight": len(live), "age": age, "recent": len(recent), "avg_ms": avg,
                "last_ms": last, "total": total}

    def all_inflight(self):
        with self.lock:
            out = {}
            for r in self.live.values():
                out[r.get("model", "?")] = out.get(r.get("model", "?"), 0) + 1
            return out


class NoEvents:
    """Stand-in when there is no llama-swap event stream: in-flight counts then come
    from each server's own running gauge, and image models show no per-image times."""

    available = False
    connected = True
    error = None

    def start(self):
        pass

    def stats(self, model, window, now):
        return {}

    def all_inflight(self):
        return {}


# ----------------------------------------------------------------------------- backends

class Backend:
    """One model server.  opts: adapter (force one), weights_gb (when the server does
    not export its weight size), kind ("llm", "image", "other" to skip detection)."""

    def __init__(self, model, proxy, window, adapters, state="?", title="", opts=None):
        self.model = model
        self.proxy = proxy.rstrip("/")
        self.window = window
        self.adapters = adapters
        self.state = state
        self.title = title
        self.opts = dict(opts or {})
        self.kind = self.opts.get("kind")  # "llm" (has an adapter), "image", "other"
        self.adapter = None   # adapter name
        self.spec = {}
        self.note = ""        # why a backend shows no token columns
        self.prev = None      # (t, metrics)
        self.cur = None
        self.err = None
        self.first = None     # metrics at tool start, for session totals
        self.samples = deque()  # (t, prompt_total, gen_total) for the windowed average
        self.gen_hist = deque(maxlen=HIST_LEN)
        self.prompt_hist = deque(maxlen=HIST_LEN)
        self.kv_hist = deque(maxlen=HIST_LEN)
        self.inflight_hist = deque(maxlen=HIST_LEN)
        self.rates = {}
        self.sse = {}
        self.last_ttft_ms = None
        self.last_itl_ms = None
        self.last_accept = None
        self.last_hit = None

    def _set_adapter(self, metrics):
        try:
            name = pick_adapter(self.adapters, metrics, self.opts.get("adapter"))
        except KeyError as e:
            self.kind, self.note = "other", f"unknown adapter {e.args[0]}"
            return
        if name is None:
            self.kind, self.note = "other", "metrics, but no adapter matches"
            return
        self.kind, self.adapter, self.spec = "llm", name, self.adapters[name]

    def _detect(self):
        try:
            self._set_adapter(parse_prom(http_get(self.proxy + "/metrics")))
        except urllib.error.HTTPError as e:
            if e.code != 404:
                self.kind, self.err = None, f"/metrics HTTP {e.code}"  # try again next sample
                return
            self.err = None
            try:
                info = json.loads(http_get(self.proxy + "/model_info"))
                self.kind = "image" if info.get("is_image_gen") else "other"
            except (urllib.error.URLError, OSError, ValueError):
                self.kind = "other"
            if self.kind == "other":
                self.note = "no metrics endpoint"
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.kind = None  # try again next sample
            self.err = str(getattr(e, "reason", e))[:60]
            return
        if self.kind:
            self.err = None

    def _blank(self):
        self.gen_hist.append(0.0)
        self.prompt_hist.append(0.0)
        self.kv_hist.append(0.0)
        self.rates = {}

    def sample(self, now, sse, window):
        if self.kind is None or (self.kind == "llm" and not self.spec):
            self._detect()
        self.sse = sse.stats(self.model, window, now)
        if self.kind != "llm":
            self._blank()
            self.inflight_hist.append(float(self.sse.get("inflight", 0)))
            return
        try:
            m = parse_prom(http_get(self.proxy + "/metrics"))
            self.err = None
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.err = str(getattr(e, "reason", e))[:60]
            self.prev = self.cur
            self.cur = None
            self._blank()
            self.inflight_hist.append(float(self.sse.get("inflight", 0)))
            return
        if self.first is None:
            self.first = m
        self.prev, self.cur = self.cur, (now, m)
        sp = self.spec
        self.samples.append((now, counter(m, sp.get("prompt_total")) or 0.0,
                             counter(m, sp.get("gen_total")) or 0.0))
        while len(self.samples) > 2 and now - self.samples[0][0] > self.window:
            self.samples.popleft()
        self._compute()
        if "inflight" in self.sse:
            self.inflight_hist.append(float(self.sse["inflight"]))
        else:
            self.inflight_hist.append(float(self.rates.get("running") or 0.0))

    def _delta(self, keys):
        """Growth of the summed counters since the previous sample, or None when the
        server does not export them."""
        if not self.prev or not self.cur:
            return None
        now_v = counter(self.cur[1], keys)
        if now_v is None:
            return None
        return now_v - (counter(self.prev[1], keys) or 0.0)

    def _ratio(self, keys, last):
        """Δa / Δb for a [a, b] counter pair, holding the last value while b is idle."""
        pair = as_list(keys)
        if len(pair) != 2:
            return last
        da, db = self._delta(pair[0]), self._delta(pair[1])
        if da is None or db is None:
            return last
        return da / db if db > 0 else last

    def _compute(self):
        if not self.prev or not self.cur:
            self.rates = {}
            return
        sp = self.spec
        dt = max(1e-3, self.cur[0] - self.prev[0])
        m = self.cur[1]
        r = {}

        # Live figures from per-step counters, then the server's gauges, then the
        # completion counters as a last resort.
        gl = self._delta(sp.get("gen_live"))
        if gl is not None:
            r["gen_live"], r["live_source"] = gl / dt, sp.get("live_label", "step counters")
        elif len(as_list(sp.get("gen_live_product"))) == 2:
            g, c = as_list(sp["gen_live_product"])

            def prod(mm):
                return (gauge(mm, g) or 0.0) * (gauge(mm, c) or 0.0)
            r["gen_live"] = max(0.0, prod(m) - prod(self.prev[1])) / dt
            r["live_source"] = sp.get("live_label", "decode calls x busy slots")
        elif gauge(m, sp.get("gen_gauge")) is not None:
            r["gen_live"] = gauge(m, sp.get("gen_gauge"))
            r["live_source"] = sp.get("fallback_label", "gauges")
        else:
            r["gen_live"] = (self._delta(sp.get("gen_total")) or 0.0) / dt
            r["live_source"] = "completion counters"
        pl = self._delta(sp.get("prompt_live"))
        if pl is None:
            pl = self._delta(sp.get("prompt_fallback"))
        if pl is None:
            pl = self._delta(sp.get("prompt_total"))
        r["prompt_live"] = (pl or 0.0) / dt
        r["prompt_cached_live"] = (self._delta(sp.get("prompt_cached_live")) or 0.0) / dt

        r["running"] = gauge(m, sp.get("running")) or 0.0
        r["queued"] = gauge(m, sp.get("queued")) or 0.0
        r["kv_usage"] = gauge(m, sp.get("kv_usage"))  # None: not exported
        hit = gauge(m, sp.get("cache_hit"))
        if hit is None and sp.get("cache_hit_counters"):
            hit = self.last_hit = self._ratio(sp["cache_hit_counters"], self.last_hit)
        r["cache_hit"] = hit
        acc = gauge(m, sp.get("accept_len"))
        if acc is None and sp.get("accept_counters"):
            per_draft = self._ratio(sp["accept_counters"], None)
            if per_draft is not None:
                self.last_accept = 1.0 + per_draft
            acc = self.last_accept
        r["accept_len"] = acc

        # Weights: the server's own figure, else the configured size.
        w = gauge(m, sp.get("weights"))
        if w is not None:
            w_bytes = w * (GIB if sp.get("weights_unit", "GiB") == "GiB" else GB)
        elif self.opts.get("weights_gb"):
            w_bytes = float(self.opts["weights_gb"]) * GB
        else:
            w_bytes = None
        r["weight_bytes"] = w_bytes

        # Windowed averages from the completion-time counters.
        t0, p0, g0 = self.samples[0]
        span = max(1e-3, self.cur[0] - t0)
        r["prompt_avg"] = (self.samples[-1][1] - p0) / span
        r["gen_avg"] = (self.samples[-1][2] - g0) / span

        # Forward passes per second, from a counter that ticks once per decode pass.
        passes = (self._delta(sp.get("passes")) or 0.0) / dt
        if passes <= 0 and r["gen_live"] > 0:
            passes = r["gen_live"] / (max(1.0, acc or 1.0) * max(1.0, r["running"]))
        r["passes_ps"] = passes

        # Bytes per pass: resident weights plus the KV cache of the running sequences.
        kv_total = gauge(m, sp.get("kv_total")) or 0.0
        kv_tokens = gauge(m, sp.get("kv_max_tokens")) or 0.0
        seq_tokens = gauge(m, sp.get("kv_seq_tokens")) or 0.0
        kv_unit = GIB if sp.get("kv_unit", "GiB") == "GiB" else GB
        kv_bytes = kv_total * kv_unit * seq_tokens / kv_tokens if kv_tokens else 0.0
        r["kv_read_bytes"] = kv_bytes
        r["gbps"] = (w_bytes + kv_bytes) / GB * passes if w_bytes is not None else None

        for key, attr in (("ttft", "last_ttft_ms"), ("itl", "last_itl_ms")):
            for base in as_list(sp.get(key)):
                c = self._delta(base + "_count")
                if c:
                    setattr(self, attr, (self._delta(base + "_sum") or 0.0) / c * 1e3)
                    break
        r["ttft_ms"] = self.last_ttft_ms
        r["itl_ms"] = self.last_itl_ms

        f = self.first or {}
        tp, tg = counter(m, sp.get("prompt_total")) or 0.0, counter(m, sp.get("gen_total")) or 0.0
        r["sess_prompt"] = tp - (counter(f, sp.get("prompt_total")) or 0.0)
        r["sess_gen"] = tg - (counter(f, sp.get("gen_total")) or 0.0)
        r["tot_prompt"], r["tot_gen"] = tp, tg
        r["tot_req"] = counter(m, sp.get("requests_total")) or 0.0
        self.rates = r
        self.gen_hist.append(r["gen_live"])
        self.prompt_hist.append(r["prompt_live"])
        self.kv_hist.append(r["kv_usage"] or 0.0)


class Sources:
    """Where backends come from: llama-swap's /running (router = "llama-swap") and the
    static [[backends]] list, with per-model overrides from [models."<name>"]."""

    def __init__(self, router, llama_swap, static, models, adapters):
        self.router = router
        self.llama_swap = llama_swap.rstrip("/")
        self.static = static      # [{"name", "url", ...opts}]
        self.models = models      # name -> opts
        self.adapters = adapters

    def _opts(self, name, extra=None):
        o = dict(self.models.get(name) or {})
        o.update({k: v for k, v in (extra or {}).items() if k not in ("name", "url")})
        return o

    def discover(self, backends, window):
        """Refresh the backend list.  Keeps history for models still listed; new ones
        are added, gone ones (an image model after its TTL) dropped."""
        seen, err = {}, None
        if self.router == "llama-swap":
            try:
                data = json.loads(http_get(self.llama_swap + "/running"))
            except (urllib.error.URLError, OSError, ValueError) as e:
                data = {}
                err = f"llama-swap unreachable: {str(getattr(e, 'reason', e))[:60]}"
            for entry in data.get("running", []):
                model = entry.get("model", "?")
                proxy = entry.get("proxy") or ""
                if not proxy:
                    continue
                b = backends.get(model) or Backend(model, proxy, window, self.adapters,
                                                   title=entry.get("name", ""), opts=self._opts(model))
                b.state = entry.get("state", "?")
                b.proxy = proxy.rstrip("/")
                seen[model] = b
        for s in self.static:
            name = s["name"]
            b = backends.get(name) or Backend(name, s["url"], window, self.adapters, state="static",
                                              opts=self._opts(name, s))
            seen[name] = b
        if not seen and not err and self.router != "llama-swap":
            err = "no backends: set router = \"llama-swap\" or add [[backends]] / --backend"
        return seen, err


# ----------------------------------------------------------------------------- CPU, RAM, GPU, processes

class SysStats:
    """CPU busy % (all cores, from /proc/stat deltas) and RAM used % (MemTotal minus
    MemAvailable).  On the GB10 the CPU and GPU share one LPDDR5X pool, so RAM used is
    the real memory-pressure figure; NVML reports GPU memory as N/A."""

    def __init__(self):
        self.prev = None

    def cpu_pct(self):
        try:
            with open("/proc/stat") as f:
                vals = [int(x) for x in f.readline().split()[1:]]
        except (OSError, ValueError):
            return None
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
        total = sum(vals[:8])  # guest time is already inside user/nice
        prev, self.prev = self.prev, (idle, total)
        if not prev or total <= prev[1]:
            return None
        return 100.0 * (1.0 - (idle - prev[0]) / (total - prev[1]))

    @staticmethod
    def ram():
        """(used %, used GiB, total GiB) or None."""
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    info[k] = int(v.split()[0])
            total, avail = info["MemTotal"], info["MemAvailable"]
        except (OSError, ValueError, KeyError):
            return None
        used = total - avail
        return 100.0 * used / total, used / 1024 ** 2, total / 1024 ** 2


def gpu_stats():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,temperature.gpu,clocks.sm,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip().split(",")
        util, power, temp, clk, name = [x.strip() for x in out[:5]]
        return {"util": float(util), "power": float(power), "temp": float(temp),
                "clock": float(clk), "name": name}
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


def gpu_procs():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        ).stdout
        procs = {}
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                procs[int(parts[0])] = float(parts[1]) if parts[1].replace(".", "").isdigit() else 0.0
        return procs
    except (subprocess.SubprocessError, ValueError):
        return {}


def ppid_map():
    m = {}
    try:
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/stat") as f:
                    stat = f.read()
                m[int(d)] = int(stat.rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                pass
    except OSError:
        pass
    return m


def listener_pid(port):
    if not shutil.which("ss"):
        return None
    try:
        out = subprocess.run(["ss", "-H", "-ltnp", f"sport = :{port}"],
                             capture_output=True, text=True, timeout=2).stdout
        m = re.search(r"pid=(\d+)", out)
        return int(m.group(1)) if m else None
    except subprocess.SubprocessError:
        return None


def backend_procs(backends):
    """model -> (pid with the most GPU memory, GPU MiB of the listener's whole tree)."""
    procs = gpu_procs()
    if not procs:
        return {}
    parents = ppid_map()
    result = {}
    for model, b in backends.items():
        m = re.search(r":(\d+)$", b.proxy)
        if not m:
            continue
        root = listener_pid(int(m.group(1)))
        if root is None:
            continue
        mib, best_pid, best_mib = 0.0, root, -1.0
        for pid, used in procs.items():
            p, hops = pid, 0
            while p and p != root and hops < 20:
                p = parents.get(p, 0)
                hops += 1
            if p == root:
                mib += used
                if used > best_mib:
                    best_pid, best_mib = pid, used
        result[model] = (best_pid, mib)
    return result


# ----------------------------------------------------------------------------- drawing

def nice_max(v, floor=10.0):
    if v <= floor:
        return floor
    for base in (20, 40, 100, 200, 400, 1000, 2000, 4000, 10000, 20000, 40000, 100000):
        if v <= base:
            return float(base)
    return float(math.ceil(v / 100000) * 100000)


def draw_chart(put, y0, x0, height, width, series, ymax, times, border_attr=0, title=None):
    """nvtop-style box chart.  series: list of (label, values, attr).  The box spans
    rows y0 .. y0+height-2, x-axis labels sit on row y0+height-1, y labels take the 5
    columns left of the box.  One column per sample, newest at the right."""
    lab_w = 5
    px0 = x0 + lab_w
    plot_w = max(10, width - lab_w - 2)
    inner_x = px0 + 1
    top, bottom = y0, y0 + height - 2
    inner_top, inner_bot = top + 1, bottom - 1
    inner_rows = inner_bot - inner_top + 1
    if inner_rows < 2:
        return
    put(top, px0, "┌" + "─" * plot_w + "┐", border_attr)
    # Title and colour-keyed legend live in the top border, so the plot stays clear.
    x = px0 + 2
    if title:
        put(top, x, f" {title} ", border_attr | curses.A_BOLD)
        x += len(title) + 3
    for label, _vals, attr in series:
        item = f" ━ {label} "
        if x + len(item) > px0 + plot_w:
            put(top, x, " … ", border_attr)
            break
        put(top, x, item, attr | curses.A_BOLD)
        x += len(item)
    for r in range(inner_top, inner_bot + 1):
        put(r, px0, "│", border_attr)
        put(r, px0 + plot_w + 1, "│", border_attr)
    put(bottom, px0, "└" + "─" * plot_w + "┘", border_attr)
    used_rows = set()
    for frac in (1.0, 0.0, 0.5, 0.75, 0.25):  # extremes first, skip collisions
        r = inner_bot - round(frac * (inner_rows - 1))
        if r in used_rows:
            continue
        used_rows.add(r)
        v = ymax * frac
        put(r, x0, f"{v:>4.0f}" if v >= 10 or v == 0 else f"{v:>4.1f}")
    tlist = list(times)
    step = (tlist[-1] - tlist[0]) / (len(tlist) - 1) if len(tlist) > 1 else 1.0
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        c = round(frac * (plot_w - 1))
        label = f"-{(plot_w - 1 - c) * step:.0f}s" if c < plot_w - 1 else "0s"
        put(bottom + 1, max(x0, min(inner_x + c - len(label) // 2, x0 + width - len(label) - 1)), label)
    for _label, vals, attr in reversed(series):  # first series ends up on top
        vals = list(vals)[-plot_w:]
        rows = [inner_bot - round(min(v, ymax) / ymax * (inner_rows - 1)) for v in vals]
        voff = plot_w - len(rows)
        prev = None
        for i, r in enumerate(rows):
            c = inner_x + voff + i
            if prev is None or prev == r:
                put(r, c, "─", attr)
            else:
                lo, hi = min(prev, r), max(prev, r)
                put(prev, c, "┐" if r > prev else "┘", attr)
                for rr in range(lo + 1, hi):
                    put(rr, c, "│", attr)
                put(r, c, "└" if r > prev else "┌", attr)
            prev = r


def bar(label, frac, width, text):
    inner = max(4, width - len(label) - 2)
    filled = int(max(0.0, min(1.0, frac)) * inner)
    body = "|" * filled + " " * (inner - filled)
    body = body[: max(0, inner - len(text) - 1)] + " " + text if len(text) < inner else text[:inner]
    return f"{label}[{body}]"


TABLE_HDR = ("PID", "MODEL", "STATE", "SRV", "RUN", "QUE", "GEN/s", "PRM/s", "GEN avg", "PRM avg",
             "ACC", "KV%", "HIT%", "TTFT", "ITL", "GB/s", "GPU MEM", "WEIGHTS")
TABLE_FMT = ("{:>7} {:<20} {:<6} {:<8} {:>3} {:>3} {:>7} {:>7} {:>7} {:>7} {:>5} {:>4} {:>4} "
             "{:>5} {:>4} {:>5} {:>9} {:>8}")


def pct(v):
    return "-" if v is None else f"{v * 100:.0f}"


def table_row(b, proc):
    r = b.rates
    s = b.sse or {}
    pid, mib = proc if proc else ("-", None)
    mem = f"{mib / 1024:.1f}GiB" if mib else "-"
    srv = (b.adapter or b.kind or "?")[:8]
    if b.err and b.kind != "llm":
        return TABLE_FMT.format(pid, b.model[:20], "down", srv, *([""] * 14)) + f"  {b.err}"
    head = f"{pid:>7} {b.model[:20]:<20} {b.state[:6]:<6} {srv:<8} {s.get('inflight', 0) if s else '-':>3}     "
    if b.kind == "image":
        if not s:
            return head + f"images: no event stream (router none)   GPU MEM {mem}"
        avg = f"{s['avg_ms'] / 1000:.1f}s" if s.get("avg_ms") else "-"
        last = f"{s['last_ms'] / 1000:.1f}s" if s.get("last_ms") else "-"
        busy = f"  busy {s['age']:.0f}s" if s.get("inflight") else ""
        return (head + f"images: {s.get('total', 0)} done this session, {s.get('recent', 0)} in window, "
                f"last {last}, avg {avg}{busy}   GPU MEM {mem}")
    if b.kind == "other":
        return head + f"({b.note or 'no metrics endpoint'})   GPU MEM {mem}"
    if b.err:
        return TABLE_FMT.format(pid, b.model[:20], "down", srv, *([""] * 14)) + f"  {b.err}"
    if not r:
        return TABLE_FMT.format(pid, b.model[:20], b.state[:6], srv, s.get("inflight", ""), "", "(wait)",
                                *([""] * 9), mem, "")
    run = s.get("inflight", int(r["running"]))
    wb = r.get("weight_bytes")
    return TABLE_FMT.format(
        pid, b.model[:20], b.state[:6], srv, run, int(r["queued"]),
        f"{r['gen_live']:.1f}", f"{r['prompt_live']:.0f}", f"{r['gen_avg']:.1f}", f"{r['prompt_avg']:.0f}",
        f"{r['accept_len']:.2f}" if r["accept_len"] else "-",
        pct(r["kv_usage"]), pct(r["cache_hit"]),
        ms(r["ttft_ms"]), ms(r["itl_ms"]),
        f"{r['gbps']:.0f}" if r["gbps"] is not None and r["passes_ps"] > 0 else "-",
        mem, f"{wb / GB:.1f}GB" if wb is not None else "-",
    )


def total_gbps(backends):
    return sum(b.rates.get("gbps") or 0.0 for b in backends.values())


def io_text(total, peak):
    if peak:
        return f"~{total:.0f}/{peak:.0f} GB/s {total / peak * 100:.0f}%"
    return f"~{total:.0f} GB/s (no peak: set peak_gbps)"


# ----------------------------------------------------------------------------- modes

def run_plain(args, backends, sse):
    sysstats = SysStats()
    sysstats.cpu_pct()  # prime the delta
    n = 0
    err = None
    while args.count <= 0 or n < args.count:
        now = time.time()
        if n % 10 == 0:
            backends, err = args.sources.discover(backends, args.window)
        for b in backends.values():
            b.sample(now, sse, args.window)
        g = gpu_stats()
        cpu = sysstats.cpu_pct()
        ram = sysstats.ram()
        stamp = datetime.now().strftime("%H:%M:%S")
        if err:
            print(f"{stamp} {err}")
        if n > 0:
            gl = f"GPU {g['util']:.0f}%  {g['power']:.0f} W  {g['temp']:.0f}°C" if g else ""
            cl = f"CPU {cpu:.0f}%" if cpu is not None else "CPU -"
            rl = f"RAM {ram[0]:.0f}% ({ram[1]:.0f}/{ram[2]:.0f} GiB)" if ram else "RAM -"
            io = total_gbps(backends)
            ev = ("ok" if sse.connected else "down") if sse.available else "none"
            print(f"{stamp} {cl}  {gl}  mem IO {io_text(io, args.peak)}  {rl}  events {ev}")
            print("         " + TABLE_FMT.format(*TABLE_HDR))
            for b in backends.values():
                print("         " + table_row(b, None))
        n += 1
        time.sleep(args.interval)
    return 0


def run_tui(stdscr, args, backends, sse):
    curses.curs_set(0)
    curses.use_default_colors()
    has_color = curses.has_colors()
    if has_color:
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_GREEN, -1)
        curses.init_pair(4, curses.COLOR_MAGENTA, -1)
        curses.init_pair(5, curses.COLOR_RED, -1)
        curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_GREEN)
        curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_CYAN)

    def cp(n):
        return curses.color_pair(n) if has_color else 0

    palette = [1, 2, 3, 4, 5]
    interval = args.interval
    tick = 0
    err = None
    times = deque(maxlen=HIST_LEN)
    gpu_hist = deque(maxlen=HIST_LEN)
    bw_hist = deque(maxlen=HIST_LEN)
    cpu_hist = deque(maxlen=HIST_LEN)
    ram_hist = deque(maxlen=HIST_LEN)
    sysstats = SysStats()
    sysstats.cpu_pct()  # prime the delta
    procs = {}
    while True:
        now = time.time()
        if tick % 10 == 0:
            backends, err = args.sources.discover(backends, args.window)
        for b in backends.values():
            b.sample(now, sse, args.window)
        g = gpu_stats()
        cpu = sysstats.cpu_pct()
        ram = sysstats.ram()
        if tick % 5 == 0:
            procs = backend_procs(backends)
        total = total_gbps(backends)
        times.append(now)
        gpu_hist.append(g["util"] if g else 0.0)
        bw_hist.append(min(100.0, total / args.peak * 100) if args.peak else 0.0)
        cpu_hist.append(cpu if cpu is not None else 0.0)
        ram_hist.append(ram[0] if ram else 0.0)
        tick += 1

        stdscr.erase()
        h, w = stdscr.getmaxyx()

        def put(y, x, s, attr=0):
            if 0 <= y < h and 0 <= x < w:
                try:
                    stdscr.addstr(y, x, s[: max(0, w - x - 1)], attr)
                except curses.error:
                    pass

        # Header.
        name = g["name"] if g else "GPU"
        put(0, 0, "Device 0 ", cp(1))
        put(0, 9, f"[{name}]", cp(2))
        x = 11 + len(name)
        if g:
            put(0, x, f"  GPU {g['clock']:.0f}MHz  TEMP {g['temp']:.0f}°C  POW {g['power']:.0f} W")
        right = f"nvtokentop {datetime.now():%H:%M:%S}  {interval:.1f}s  q quit +/- interval"
        put(0, max(x + 44, w - len(right) - 1), right, curses.A_DIM)
        inflight = sse.all_inflight()
        n_inflight = sum(inflight.values())
        if not sse.available:
            n_inflight = int(sum(b.rates.get("running") or 0 for b in backends.values()))
        kv = max([b.rates.get("kv_usage") or 0.0 for b in backends.values()] or [0.0])
        half = max(24, (w - 2) // 2)
        put(1, 0, bar("CPU", (cpu or 0) / 100, half, f"{cpu:.0f}%" if cpu is not None else "n/a"), cp(4))
        put(1, half + 1, bar("GPU", (g["util"] / 100) if g else 0, w - half - 2,
                             f"{g['util']:.0f}%" if g else "n/a"), cp(1))
        put(2, 0, bar("MEM IO", total / args.peak if args.peak else 0, half,
                      io_text(total, args.peak)), cp(3))
        put(2, half + 1, bar("RAM", (ram[0] / 100) if ram else 0, w - half - 2,
                             f"{ram[1]:.1f}/{ram[2]:.1f}Gi {ram[0]:.0f}%" if ram else "n/a"), cp(2))
        status = f"KV {kv * 100:.0f}%  {n_inflight} req in flight"
        if err:
            put(3, 0, err, cp(5))
        elif not sse.connected:
            put(3, 0, f"llama-swap event stream down ({sse.error or 'connecting'}): in-flight and image "
                      f"counts stale", cp(5))
        else:
            put(3, 0, status, curses.A_DIM)

        # Charts, in priority order, as many as fit.
        llms = [b for b in backends.values() if b.kind == "llm"]
        charts = []
        def short(b):
            return b.model.split("/")[-1]

        s1 = [(short(b), b.gen_hist, cp(palette[i % len(palette)])) for i, b in enumerate(llms)]
        charts.append(("generation tok/s", s1, None))
        s2 = [(short(b), b.prompt_hist, cp(palette[i % len(palette)])) for i, b in enumerate(llms)]
        s3 = [("CPU", cpu_hist, cp(4)), ("GPU", gpu_hist, cp(1)),
              ("mem IO", bw_hist, cp(3)), ("RAM", ram_hist, cp(2))]
        for b in llms:
            if b.rates.get("kv_usage") is None:
                continue
            s3.append((f"{short(b)} KV", deque([v * 100 for v in b.kv_hist]), cp(5)))
        charts.append(("utilization %", s3, 100.0))
        charts.append(("prompt tok/s", s2, None))
        s4 = [(short(b) + (" images" if b.kind == "image" else ""), b.inflight_hist,
               cp(palette[i % len(palette)])) for i, b in enumerate(backends.values())]
        charts.append(("requests in flight", s4, None))

        table_rows = 1 + max(1, len(backends)) + 1
        chart_top = 4
        avail = h - chart_top - table_rows - 1
        n_charts = max(0, min(len(charts), avail // 7))
        y = chart_top
        if n_charts:
            ch_h = avail // n_charts
            plot_w = max(10, w - 7)
            for title, series, ymax in charts[:n_charts]:
                if ymax is None:
                    peak = max([max(list(s[1])[-plot_w:] or [0]) for s in series] or [0])
                    ymax = nice_max(peak, floor=4.0 if title.startswith("requests") else 10.0)
                draw_chart(put, y, 0, ch_h, w, series, ymax, times, title=title)
                y += ch_h
        else:
            put(y, 0, "(terminal too short for charts)", curses.A_DIM)
            y += 1

        # Table.
        put(y, 0, TABLE_FMT.format(*TABLE_HDR).ljust(w - 1), cp(6) if has_color else curses.A_REVERSE)
        y += 1
        if not backends:
            put(y, 0, "no models running" + (" behind llama-swap" if args.sources.router == "llama-swap"
                                             else ""))
            y += 1
        for i, b in enumerate(backends.values()):
            put(y, 0, table_row(b, procs.get(b.model)), cp(7) if (has_color and i == 0) else 0)
            y += 1
        sess_p = sum(b.rates.get("sess_prompt", 0) for b in backends.values())
        sess_g = sum(b.rates.get("sess_gen", 0) for b in backends.values())
        tot_p = sum(b.rates.get("tot_prompt", 0) for b in backends.values())
        tot_g = sum(b.rates.get("tot_gen", 0) for b in backends.values())
        src = ", ".join(sorted({b.rates["live_source"] for b in llms if b.rates})) or "-"
        put(y, 0, f"tokens this session: in {fmt_int(sess_p)}  out {fmt_int(sess_g)}   since server start: "
                  f"in {fmt_int(tot_p)}  out {fmt_int(tot_g)}   live columns from {src}; avg over "
                  f"{args.window:.0f}s of completed requests; GB/s = (weights + running KV) x passes/s",
            curses.A_DIM)
        stdscr.refresh()

        stdscr.timeout(int(interval * 1000))
        try:
            ch = stdscr.getch()
        except KeyboardInterrupt:
            return 0
        if ch in (ord("q"), ord("Q"), 27):
            return 0
        if ch in (ord("+"), ord("=")):
            interval = min(30.0, interval + 0.5)
        elif ch in (ord("-"), ord("_")):
            interval = max(0.5, interval - 0.5)


def parse_backend_arg(text):
    """NAME=URL[,key=value...] -> backend dict (adapter=, weights_gb=, kind=)."""
    name, _, rest = text.partition("=")
    url, *extras = rest.split(",")
    if not name or not url:
        sys.exit(f"nvtokentop: --backend wants NAME=URL[,adapter=X,weights_gb=N], got {text!r}")
    d = {"name": name, "url": url}
    for e in extras:
        k, _, v = e.partition("=")
        d[k] = float(v) if k == "weights_gb" else v
    return d


def main():
    ap = argparse.ArgumentParser(description="nvtop for tokens: live activity of local model servers")
    ap.add_argument("--config", help="config file (TOML, or JSON by extension)")
    ap.add_argument("--router", choices=("llama-swap", "none"), help="where models are listed (default llama-swap)")
    ap.add_argument("--llama-swap", help="llama-swap base URL (default http://127.0.0.1:8000)")
    ap.add_argument("--backend", action="append", default=[], metavar="NAME=URL[,adapter=X,weights_gb=N]",
                    help="a model server to watch directly; repeatable")
    ap.add_argument("-i", "--interval", type=float, help="seconds between samples (default 1)")
    ap.add_argument("-w", "--window", type=float, help="seconds for the avg columns (default 30)")
    ap.add_argument("--peak", type=float, help="memory bandwidth spec, GB/s (default: from the GPU name)")
    ap.add_argument("--plain", action="store_true", help="print blocks instead of the curses TUI")
    ap.add_argument("-n", "--count", type=int, default=0, help="with --plain: stop after N samples")
    ap.add_argument("--list-adapters", action="store_true", help="print the metric adapters as TOML and exit")
    ap.add_argument("--version", action="version", version=f"nvtokentop {__version__}")
    args = ap.parse_args()

    cfg, cfg_path = load_config(args.config)
    adapters = load_adapters(cfg)
    if args.list_adapters:
        print_adapters(adapters)
        return 0
    router = args.router or cfg.get("router", "llama-swap")
    llama_swap = args.llama_swap or cfg.get("llama_swap", "http://127.0.0.1:8000")
    args.interval = args.interval or float(cfg.get("interval", 1.0))
    args.window = args.window or float(cfg.get("window", 30.0))
    static = [dict(b) for b in cfg.get("backends", [])] + [parse_backend_arg(t) for t in args.backend]
    for b in static:
        if not b.get("name") or not b.get("url"):
            sys.exit(f"nvtokentop: every backend needs a name and a url ({cfg_path or '--backend'})")
    args.sources = Sources(router, llama_swap, static, cfg.get("models") or {}, adapters)

    if args.peak is None:
        args.peak = float(cfg["peak_gbps"]) if "peak_gbps" in cfg else peak_for((gpu_stats() or {}).get("name"))[0]

    backends = {}
    sse = Inflight(llama_swap) if router == "llama-swap" else NoEvents()
    sse.start()
    if args.plain or not sys.stdout.isatty():
        try:
            return run_plain(args, backends, sse)
        except KeyboardInterrupt:
            return 0
    return curses.wrapper(run_tui, args, backends, sse)


if __name__ == "__main__":
    sys.exit(main())
