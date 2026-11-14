#!/usr/bin/env python3
"""ailab-bench: the same speed test for every model.

Runs on the box, standard library only. Talks to an OpenAI-compatible server (vLLM).

  python3 ~/ailab/bench/ailab-bench.py --label qwen38-27b-baseline \
      --base-url http://TAILSCALE_IP:8000/v1 --model qwen3.8-27b

Measures:
  - single-user decode tok/s on a code prompt and a prose prompt (median of --repeats)
  - time to first token (and prefill tok/s) on a ~30k-token prompt; a random prefix
    on every request defeats the prefix cache, so each run is a real prefill
  - total tok/s with --concurrency requests at once
  - tool-call check (--tool-tries tries; the reply must be a get_weather call for Pune)
  - during the run: lowest MemAvailable, GPU temperature/clock/power, NV_ERR_NO_MEMORY
    lines in the kernel log, speculative-decoding acceptance (from /metrics)

Writes ~/ailab/bench/results/<label>-<timestamp>.json and prints a summary.
Sampling is fixed for every model: temperature 0.6, top_p 0.95, thinking off
(chat_template_kwargs.enable_thinking=false; templates that don't know it ignore it).
"""
import argparse
import glob
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

CODE_PROMPT = (
    "Write a Python module that implements a thread-safe LRU cache with per-entry TTL "
    "expiry. Include type hints, docstrings, and a set of pytest unit tests. "
    "Output only the code."
)
PROSE_PROMPT = (
    "Write a 600-word essay on how the printing press changed European society between "
    "1450 and 1600. Plain prose, no headings, no lists."
)
TOOL_PROMPT = "What's the weather in Pune right now? Use the tool."
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]


def load_key(path):
    with open(os.path.expanduser(path)) as f:
        for line in f:
            if line.startswith("VLLM_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit(f"no VLLM_API_KEY in {path}")


def nonce():
    return "Request id %016x.\n\n" % random.getrandbits(64)


class Client:
    def __init__(self, base, key, model, extra):
        self.base, self.key, self.model, self.extra = base.rstrip("/"), key, model, extra

    def body(self, messages, max_tokens, **kw):
        b = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        b.update(kw)
        b.update(self.extra)
        return b

    def _open(self, body, timeout=900):
        req = urllib.request.Request(
            self.base + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=timeout)

    def stream(self, prompt, max_tokens):
        """One streamed request. Returns timing and token counts."""
        body = self.body([{"role": "user", "content": prompt}], max_tokens,
                         stream=True, stream_options={"include_usage": True})
        t0 = time.perf_counter()
        t_first = t_last = None
        usage, reasoning_seen, finish = None, False, None
        with self._open(body) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    d = ch.get("delta") or {}
                    r_txt = d.get("reasoning_content") or d.get("reasoning")
                    if d.get("content") or r_txt:
                        now = time.perf_counter()
                        t_first = t_first or now
                        t_last = now
                    if r_txt:
                        reasoning_seen = True
                    finish = ch.get("finish_reason") or finish
        t_end = time.perf_counter()
        comp = (usage or {}).get("completion_tokens", 0)
        prompt_toks = (usage or {}).get("prompt_tokens", 0)
        ttft = (t_first - t0) if t_first else None
        decode = (comp - 1) / (t_last - t_first) if t_first and t_last > t_first and comp > 1 else None
        return {
            "prompt_tokens": prompt_toks, "completion_tokens": comp, "ttft_s": ttft,
            "decode_tps": decode, "total_s": t_end - t0, "reasoning_seen": reasoning_seen,
            "finish_reason": finish,
        }

    def tool_call(self):
        body = self.body([{"role": "user", "content": TOOL_PROMPT}], 512,
                         tools=TOOLS, tool_choice="auto")
        try:
            with self._open(body, timeout=300) as r:
                msg = json.load(r)["choices"][0]["message"]
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "why": f"request failed: {e}"}
        calls = msg.get("tool_calls") or []
        if not calls:
            return {"ok": False, "why": "no tool call; content=" + (msg.get("content") or "")[:120]}
        fn = calls[0].get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            return {"ok": False, "why": "arguments not JSON: " + str(fn.get("arguments"))[:120]}
        ok = fn.get("name") == "get_weather" and "pune" in str(args.get("city", "")).lower()
        return {"ok": ok, "why": "" if ok else f"got {fn.get('name')}({args})"}


class Sampler(threading.Thread):
    """Samples memory and GPU every 2 s while the benchmark runs."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.min_avail_gib = None
        self.temps, self.clocks, self.power = [], [], []

    @staticmethod
    def mem_available_gib():
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576
        return None

    def run(self):
        while not self.stop.is_set():
            a = self.mem_available_gib()
            if a is not None:
                self.min_avail_gib = a if self.min_avail_gib is None else min(self.min_avail_gib, a)
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().split(",")
                for vals, raw in zip((self.temps, self.clocks, self.power), out):
                    try:
                        vals.append(float(raw))
                    except ValueError:
                        pass
            except Exception:  # noqa: BLE001
                pass
            self.stop.wait(2)


def nv_err_count():
    try:
        out = subprocess.run(["journalctl", "-k", "-b", "--no-pager", "-q", "--grep", "NV_ERR_NO_MEMORY"],
                             capture_output=True, text=True, timeout=30).stdout
        return sum(1 for line in out.splitlines() if "NV_ERR_NO_MEMORY" in line)
    except Exception:  # noqa: BLE001
        return None


def metrics(base):
    """Requests running and speculative-decoding counters from vLLM's /metrics (no key needed)."""
    root = re.sub(r"/v1/?$", "", base.rstrip("/"))
    try:
        with urllib.request.urlopen(root + "/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception:  # noqa: BLE001
        return {}
    m = {}
    for name in ("num_requests_running", "num_requests_waiting",
                 "spec_decode_num_drafts_total", "spec_decode_num_accepted_tokens_total"):
        vals = re.findall(r"^vllm:%s\{[^}]*\} ([0-9.eE+-]+)$" % name, text, re.M)
        if vals:
            m[name] = sum(float(v) for v in vals)
    return m


def long_prompt(target_chars):
    """~30k tokens of real Python source from the box's standard library."""
    parts, n = [], 0
    for path in sorted(glob.glob("/usr/lib/python3.12/*.py")):
        try:
            with open(path, encoding="utf-8") as f:
                txt = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        parts.append(f"# ===== {os.path.basename(path)} =====\n{txt}")
        n += len(txt)
        if n >= target_chars:
            break
    src = "".join(parts)[:target_chars]
    return src + "\n\nIn three sentences, what do the modules above have in common?"


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--base-url", default="http://TAILSCALE_IP:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--key-file", default="~/ailab/vllm/env")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--long-chars", type=int, default=110000, help="~30k tokens")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--tool-tries", type=int, default=3)
    ap.add_argument("--extra", default="{}", help="JSON merged into every request")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()

    c = Client(a.base_url, load_key(a.key_file), a.model, json.loads(a.extra))
    res = {"label": a.label, "model": a.model, "base_url": a.base_url,
           "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "notes": a.notes,
           "settings": {"temperature": 0.6, "top_p": 0.95, "thinking": False, "extra": c.extra,
                        "max_tokens": a.max_tokens, "repeats": a.repeats}}

    # Wait (up to 2 min) until nobody else is using the server, so the numbers are clean.
    for _ in range(24):
        m = metrics(a.base_url)
        if m.get("num_requests_running", 0) + m.get("num_requests_waiting", 0) == 0:
            break
        print("server busy with other requests, waiting...", flush=True)
        time.sleep(5)
    res["busy_at_start"] = bool(m.get("num_requests_running", 0))

    nv0, m0 = nv_err_count(), metrics(a.base_url)
    s = Sampler()
    s.start()
    t_start = time.perf_counter()

    print("warm-up...", flush=True)
    c.stream("Say hello in five words.", 32)

    print("single user, code and prose...", flush=True)
    single = {"code": [], "prose": []}
    for i in range(a.repeats):
        for kind, p in (("code", CODE_PROMPT), ("prose", PROSE_PROMPT)):
            r = c.stream(nonce() + p, a.max_tokens)
            single[kind].append(r)
            print(f"  {kind} {i + 1}: {r['completion_tokens']} tok, {r['decode_tps'] or 0:.1f} tok/s, "
                  f"ttft {r['ttft_s'] or 0:.2f}s", flush=True)

    print("long prompt (prefill)...", flush=True)
    lp = long_prompt(a.long_chars)
    longs = []
    for i in range(2):
        r = c.stream(nonce() + lp, 128)
        longs.append(r)
        print(f"  long {i + 1}: {r['prompt_tokens']} prompt tok, ttft {r['ttft_s'] or 0:.1f}s", flush=True)

    print(f"{a.concurrency} at once...", flush=True)
    conc = [None] * a.concurrency

    def worker(i):
        conc[i] = c.stream(nonce() + (CODE_PROMPT if i % 2 == 0 else PROSE_PROMPT), a.max_tokens)

    t0 = time.perf_counter()
    th = [threading.Thread(target=worker, args=(i,)) for i in range(a.concurrency)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    conc_wall = time.perf_counter() - t0
    conc_tokens = sum(r["completion_tokens"] for r in conc if r)
    print(f"  {conc_tokens} tok in {conc_wall:.1f}s", flush=True)

    print("tool calls...", flush=True)
    tools = [c.tool_call() for _ in range(a.tool_tries)]

    s.stop.set()
    s.join(timeout=5)
    nv1, m1 = nv_err_count(), metrics(a.base_url)

    drafts = m1.get("spec_decode_num_drafts_total", 0) - m0.get("spec_decode_num_drafts_total", 0)
    acc = m1.get("spec_decode_num_accepted_tokens_total", 0) - m0.get("spec_decode_num_accepted_tokens_total", 0)
    summary = {
        "code_tps": med(r["decode_tps"] for r in single["code"]),
        "prose_tps": med(r["decode_tps"] for r in single["prose"]),
        "long_prompt_tokens": longs[-1]["prompt_tokens"],
        "long_ttft_s": med(r["ttft_s"] for r in longs),
        "prefill_tps": med(r["prompt_tokens"] / r["ttft_s"] for r in longs if r["ttft_s"]),
        "concurrent_n": a.concurrency,
        "concurrent_total_tps": round(conc_tokens / conc_wall, 1),
        "concurrent_per_user_tps": med(r["decode_tps"] for r in conc if r),
        "tool_calls_ok": f"{sum(t['ok'] for t in tools)}/{len(tools)}",
        "spec_accept_per_draft": round(acc / drafts, 2) if drafts else None,
        "min_mem_available_gib": round(s.min_avail_gib, 1) if s.min_avail_gib else None,
        "gpu_temp_max_c": max(s.temps) if s.temps else None,
        "gpu_clock_median_mhz": med(s.clocks),
        "gpu_power_max_w": max(s.power) if s.power else None,
        "nv_err_new": (nv1 - nv0) if None not in (nv0, nv1) else None,
        "reasoning_seen": any(r["reasoning_seen"] for r in single["code"] + single["prose"]),
        "bench_minutes": round((time.perf_counter() - t_start) / 60, 1),
    }
    res.update(summary=summary, single=single, long=longs, concurrent=conc, tools=tools)

    out_dir = os.path.expanduser("~/ailab/bench/results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{a.label}-{time.strftime('%Y%m%d-%H%M')}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)

    print("\n=== " + a.label + " ===")
    for k, v in summary.items():
        print(f"  {k:26} {v}")
    for t in tools:
        if not t["ok"]:
            print("  tool-call failure:", t["why"])
    print("saved", path)


if __name__ == "__main__":
    main()
