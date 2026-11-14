# home-ai-lab

Scripts that run two vLLM models on an NVIDIA GB10 machine (DGX Spark or similar) and keep it running without anyone at the box. The setup and its measurements are in [Building a home AI lab on NVIDIA GB10: setup, fail-safes and benchmarks](https://runtimenotes.com/blog/building-a-home-ai-lab-on-nvidia-gb10/).

## Tested on

| Component | Version |
|---|---|
| Hardware | MSI EdgeXpert MS-C931 (NVIDIA GB10) |
| OS | NVIDIA DGX OS |
| vLLM | 0.31 in `nvcr.io/nvidia/vllm:26.10.pre1-py3` |
| Models | Qwen3.8-27B NVFP4 with a DFlash2 draft model, Qwen3.6-35B-A3B NVFP4 |
| Chat | Open WebUI v0.13.3 |
| Remote access | Tailscale |

## What's in it

### Fail-safes

| File | What it does |
|---|---|
| [`safety/install.sh`](safety/install.sh) | Installs the fail-safes and the first model service |
| [`safety/ailab-memwatch`](safety/ailab-memwatch) | Memory watchdog |
| [`safety/ailab-memwatch.service`](safety/ailab-memwatch.service) | Runs the watchdog under systemd |
| [`safety/vllm-precheck.sh`](safety/vllm-precheck.sh) | Checks memory and state before a model starts |

### Model services

| File | What it does |
|---|---|
| [`vllm/qwen38-27b.sh`](vllm/qwen38-27b.sh) | Starts Qwen3.8-27B in vLLM |
| [`vllm/qwen36-35b.sh`](vllm/qwen36-35b.sh) | Starts Qwen3.6-35B-A3B in vLLM |
| [`safety/ailab-vllm.service`](safety/ailab-vllm.service) | Runs Qwen3.8-27B under systemd |
| [`safety/ailab-vllm-qwen36.service`](safety/ailab-vllm-qwen36.service) | Runs Qwen3.6-35B-A3B under systemd, after Qwen3.8-27B is serving |
| [`safety/install-two-models.sh`](safety/install-two-models.sh) | Adds the Qwen3.6-35B-A3B service. `--undo` goes back to Qwen3.8-27B alone. |
| [`open-webui/run.sh`](open-webui/run.sh) | Starts Open WebUI |

### Monitoring

| File | What it does |
|---|---|
| [`bin/ailab-status`](bin/ailab-status) | Shows memory, GPU, services, both models and health counters. Read-only. |

### Benchmarking

| File | What it does |
|---|---|
| [`bench/ailab-bench.py`](bench/ailab-bench.py) | Measures decode speed, time to first token, concurrent requests and tool calls |
| [`bench/g-run.sh`](bench/g-run.sh) | Starts a test model and benchmarks it |
| [`bench/run-qwen38.sh`](bench/run-qwen38.sh) | Test container for Qwen3.8-27B |
| [`bench/run-qwen36.sh`](bench/run-qwen36.sh) | Test container for Qwen3.6-35B-A3B |
| [`bench/run-laguna.sh`](bench/run-laguna.sh) | Test container for Laguna S 2.1 |

## How the fail-safes work

`install.sh` makes these changes to the system:

| Area | Change |
|---|---|
| Sleep | Masks sleep, suspend and hibernate |
| Kernel panic | Reboots 10 seconds after a panic |
| Hangs | Arms the hardware watchdog, which reboots the box after 30 seconds without a heartbeat |
| Out of memory | Sets `OOMScoreAdjust=-900` on SSH and Tailscale |
| Memory watchdog | Installs and starts `ailab-memwatch` |
| Model service | Runs Qwen3.8-27B under systemd, with at most three start attempts an hour |
| Open WebUI | Recreates the container with log size limits |

The memory watchdog reads available memory every second:

| Available memory | Action |
|---|---|
| Below 15 GiB | Logs a warning |
| Below 6 GiB for 5 seconds | Saves the model logs, stops the model containers and writes `/run/ailab-memwatch.emergency` |

While that marker exists, the start check won't start a model. A reboot clears it.

Before each model starts, `vllm-precheck.sh` checks that there's no emergency marker, that the Tailscale address is up and that enough memory is available. If the other model is still loading, it waits.

`bench/g-run.sh` stops its test model if available memory drops below 15 GiB.

## Install

### Clone and fill in the placeholders

```bash
git clone https://github.com/runtimenoteslabs/home-ai-lab.git ~/ailab
```

The scripts expect to live in `~/ailab`. Replace the two placeholders:

| Placeholder | Replace with |
|---|---|
| `TAILSCALE_IP` | The box's Tailscale address, from `tailscale ip -4` |
| `YOUR_USER` | The Linux user that runs the models |

```bash
cd ~/ailab
grep -rl --exclude-dir=.git --exclude=README.md -e TAILSCALE_IP -e YOUR_USER . \
  | xargs sed -i -e "s/TAILSCALE_IP/$(tailscale ip -4)/g" -e "s/YOUR_USER/$USER/g"
```

### Add the keys

1. Create `~/ailab/vllm/env` containing `VLLM_API_KEY=<a long random key>`. Clients send this key.
2. Create `~/ailab/open-webui/env` with Open WebUI's settings, including `WEBUI_SECRET_KEY`.
3. Run `chmod 600` on both files.

### Download the models

Download each model to `~/models` at the revision these scripts were tested with:

| Folder in `~/models` | Hugging Face repository | Revision |
|---|---|---|
| `Qwen3.8-27B-NVFP4` | [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) | `482ca0f3832238542f8f5295dde86b5f22711d80` |
| `Qwen3.8-27B-DFlash2` | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| `Qwen3.6-35B-A3B-NVFP4` | [unsloth/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) | `739af1e7aac320af1682ed1e0cce369af4c5265d` |
| `Laguna-S-2.1-NVFP4` | [poolside/Laguna-S-2.1-NVFP4](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4) | `826aacdf6d8b2699d4e367def6f17c83b06044c2` |
| `Laguna-S-2.1-DFlash-NVFP4` | [poolside/Laguna-S-2.1-DFlash-NVFP4](https://huggingface.co/poolside/Laguna-S-2.1-DFlash-NVFP4) | `b3b5921a900b9e0a1e27e50bdaeb480692a6d19b` |

For example:

```bash
hf download nvidia/Qwen3.8-27B-NVFP4 \
  --revision 482ca0f3832238542f8f5295dde86b5f22711d80 \
  --local-dir ~/models/Qwen3.8-27B-NVFP4
```

### Run the installers

The installers run as root and make the system changes listed in [How the fail-safes work](#how-the-fail-safes-work). Read them before you run them.

```bash
sudo bash ~/ailab/safety/install.sh
sudo bash ~/ailab/safety/install-two-models.sh
mkdir -p ~/.local/bin
ln -s ~/ailab/bin/ailab-status ~/.local/bin/ailab-status
```

Both models are serving about ten minutes later.

To run `ssh <host> ailab-status` from another machine, add `~/.local/bin` to `PATH` at the top of `~/.bashrc`. Ubuntu's `~/.bashrc` returns early for non-interactive shells.

## Settings to measure on your box

These values were measured on one box running these two models. Before you rely on them, check vLLM's startup log and `MemAvailable` in `/proc/meminfo` with your models serving.

| Setting | File | Value |
|---|---|---|
| Memory share | `MEM` in [`vllm/`](vllm) scripts | 0.34 and 0.31 |
| Context length | `MAXLEN` in [`vllm/`](vllm) scripts | 131,072 tokens |
| Start check | [`vllm-precheck.sh`](safety/vllm-precheck.sh) arguments in the model units | 65 GiB and 62 GiB |
| Watchdog thresholds | `MIN_AVAIL_GIB`, `WARN_AVAIL_GIB` and `DEBOUNCE` in [`safety/ailab-memwatch`](safety/ailab-memwatch) | 6 GiB, 15 GiB and 5 seconds |
| Benchmark guard | [`bench/g-run.sh`](bench/g-run.sh) | 15 GiB |

## Everyday commands

| Task | Command |
|---|---|
| Watch the status | `ailab-status -w` |
| Check the services | `systemctl status ailab-vllm ailab-vllm-qwen36 ailab-memwatch` |
| Follow the watchdog log | `journalctl -u ailab-memwatch -f` |
| Follow a model's log | `docker logs -f vllm-qwen36` or `docker logs -f vllm-qwen38-27b` |
| Stop both models | `sudo systemctl stop ailab-vllm-qwen36 ailab-vllm` |
| Start both models | `sudo systemctl start ailab-vllm ailab-vllm-qwen36` |
| Clear an emergency stop | `sudo rm /run/ailab-memwatch.emergency`, then start both models |

## License

MIT. See [LICENSE](LICENSE).
