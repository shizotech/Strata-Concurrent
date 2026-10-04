# Strata - the details

The technical side of Strata: every measured number, the API, images, all settings and how the engine works.
New here? Start with the [README](../README.md) - it has everything you need to install and use it.

> **On this page:** [Speed](#speed-measured) · [Other GPUs](#other-gpus-estimated) · [Which model?](#which-model) ·
> [Requirements](#before-you-start) · [Windows](#windows) · [Linux](#linux) · [API](#using-it) ·
> [MCP tools](#tools-from-mcp-servers) · [Images](#images-vision) ·
> [Troubleshooting](#troubleshooting) · [How it works](#how-it-works)

---

## Speed (measured)

RTX 5070 **12 GB**, Ryzen 5 7600 (6 cores), 64 GB DDR5-5200, Windows, engine 0.1.26 with the settings setup writes
(`--prefill auto`, 8-bit KV above 4K, KV streaming from 64K). One code-agent prompt per length, 256 generated tokens,
MTP speculative decoding on. "262K" is the model's full context window (a 259,943-token prompt). The IQ2_XS row was
measured with Swift 1.5's IQ2_XS, which runs at the original's speed.

### Prompt processing (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 536 | 1,299 | 2,171 | 2,126 | 2,107 | 1,304† |
| **IQ2_XS** | 534 | 1,256 | 2,092 | 1,754 | 1,752 | 1,181*† |
| **IQ3_XXS** | 482 | 1,007 | 1,745 | 1,609 | 1,602 | - |
| **IQ3_S** | 427 | 913 | 1,624 | 1,640 | 1,443 | - |
| **Coder** | 656 | 1,583 | 2,177 | 2,236 | 2,208 | 1,034** |

Engine 0.1.26; `bench/results/2026-09-29-speed-0126`. At 32K-128K that is 8-28% faster than 0.1.22. † not measured
again: 0.1.22. \* measured with images on (the image encoder's VRAM reserve leaves fewer experts cached). \*\* not
measured again: 0.1.14.

### Output (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 87.3 | 93.0 | 81.8 | 76.2 | 73.7 | 60.3† |
| **IQ2_XS** | 79.6 | 78.6 | 76.3 | 63.7 | 62.7 | 52.8† |
| **IQ3_XXS** | 61.9 | 61.6 | 58.5 | 57.2 | 49.0 | - |
| **IQ3_S** | 52.4 | 53.3 | 48.3 | 46.3 | 45.5 | - |
| **Coder** | 58.9 | 55.1 | 54.9 | 53.2 | 43.0 | 42.8† |

Engine 0.1.26, the same runs. † not measured again: 0.1.14.

Output speed depends on the text as well: speculative decoding runs faster when more of the drafted tokens are
accepted, so a different answer to the same prompt moves it by several percent. Run back to back on the 4K prompt,
0.1.14 writes 88.5 tokens/s and 0.1.12 85.7. The numbers before 0.1.13 (prompts about half as fast):
[`bench/results/2026-09-24-final`](../bench/results/2026-09-24-final/matrix.md); these:
[`bench/results/2026-09-28-speed-0114`](../bench/results/2026-09-28-speed-0114/README.md).

IQ3_XXS and IQ3_S at 262K are not measured: with their 43 / 50 GB of experts, a 260K-token context brings a 64 GB PC
to its memory limit. Use up to 128K with them on 64 GB (setup caps it). IQ3_S (engine 0.1.4 or newer) is only published
for the original model, not for Swift 1.5.

**KV streaming (engine 0.1.5):** at 64K and more, setup keeps the context's KV cache in RAM and only the part the
attention reads in VRAM (`--kv-resident 32768`), so more experts fit on the GPU. Q2_0 at 262K: 50.9 -> 62.6 tokens/s
(1,589 -> 3,872 experts in VRAM); at 128K about +6%. The attention reads exactly the same values (only where the KV lives changes); it
costs ~13.7 KB of RAM per context token (1.7 GB at 128K). Existing installs: run `START-HERE.bat --setup` once to turn
it on.

**4-bit KV cache (engine 0.1.8, optional):** `START-HERE.bat --setup` asks above 8K context (or pass `--kv q4_0`). It
halves the KV cache's memory with a Hadamard rotation before 4-bit rounding (PR #21), about 4% faster at 128K, but it
is measurably less precise on long documents (perplexity +8-12%; needle tests still pass). 8-bit stays the default.
Details: [`bench/results/2026-09-27-kv-q4`](../bench/results/2026-09-27-kv-q4/README.md).

**Hybrid K8V4 KV cache (engine 0.1.25, optional, PR #120):** `--kv k8v4` (`START-HERE.bat --setup --kv k8v4`) keeps
the keys at 8 bits and stores the values as rotated 4-bit: 23% less KV memory than 8-bit, so more experts fit in
VRAM. RTX 3090, the Coder at 198K context: 99 instead of 85 tokens/s output, the same needle results, prompts 2-5%
slower. It does not stream its KV cache (KV streaming is on by default from 64K), so it pays off mostly on large
cards at long contexts.

**Reproducible greedy output (0.1.30, opt-in, `STRATA_IQ_MT_MIN=1`):** with the IQ models, the CPU computes an
expert for one token with ggml's dot product and for several tokens with Strata's multi-token kernels, which round
slightly differently. How many tokens share an expert depends on the drafts in a verify window, so the same prompt
at temperature 0 can end in a different (equally good) answer when the drafting, the cache state or a resumed
conversation differ (issue #152). `STRATA_IQ_MT_MIN=1` (in the config's `env`) uses the multi-token kernels for
every group: the answer then no longer depends on the drafting. Measured on a Ryzen 7600 (AVX-512): IQ3_S decode
-1..-3%, the other models the same; the default stays the fastest rule.

**The draft layer's tokens (0.1.27, `--draft-vocab`):** the MTP draft layer can only propose tokens from a subset
of the vocabulary (`mtp/rt/draft_vocab.bin`). Since 0.1.27 the subset includes every Chinese, Japanese and Korean
token (106,299 ids), so answers in those languages are 15-38% faster (Q2_0, RTX 5070). Its head takes ~180 MiB of
VRAM, which the expert cache leaves free for it (0.1.28). `START-HERE.bat --setup --draft-vocab en` keeps the
English/code subset from before (40,525 ids, ~110 MiB less VRAM, English answers 1-2% faster; CJK answers get
almost no drafts). `tools/draft_vocab.py` builds and inspects subsets.

**Low-RAM mode (engine 0.1.26, chosen by setup):** normally all of a model's experts are copied into RAM (23-50 GB,
pinned) and the GPU holds a copy of the most-used ones. On a PC whose RAM cannot hold them beside the system (the
experts plus ~10 GB), setup instead maps them from one file in the model's folder (`--mmap-experts`, the pack's
`experts.bin`, +23-50 GB of disk). The OS file cache holds what the GPU does not, and it can give that memory back.
On the Coder the engine's committed memory drops from 36 to ~13 GB, with the same answers. With a big GPU (an RTX
5090 holds all of the Coder's experts, most of Q2_0's) it runs at nearly the usual speed. With a small one, most
experts come from the SSD and it is much slower (setup says so). `START-HERE.bat --setup --low-ram on|off` overrides
the choice.

**Low-RAM mode, resident (engine 0.1.30):** when the experts the GPU does not hold fit the RAM (with the same ~10 GB
beside them), setup picks the resident variant instead (`--resident-experts`): at start the engine copies exactly those
experts from `experts.bin` into RAM (page-locked when the driver allows, else locked in RAM), so while it answers
nothing is read from the SSD, however little RAM the OS leaves for its file cache. Examples with setup's context: a
32 GB PC with a 24 GB GPU runs Q2_0, IQ2_XS and the Coder this way (~16-18 GB of experts in RAM, the GPU holds the
other ~18 GB), a 32 GB PC with a 12-16 GB GPU the Coder; IQ3_XXS on a 32 GB PC stays mapped. The details:
- The prompt path borrows room in the GPU's expert cache for its buffers and puts those experts back after the prompt;
  as far as the RAM allows, their experts are kept in RAM too (so a prompt reads nothing from the SSD either).
- The cache still follows the conversation (`--adapt-every`): a swap copies the evicted expert back from VRAM into the
  RAM place of the one that replaces it, so the RAM copy keeps holding exactly what the GPU does not.
- The answers are the plain mapped mode's for the same expert placement: the bytes are the file's. With a page-locked
  copy the GPU also takes its usual share of the misses over PCIe (`--pcie-frac`), as with enough RAM; `--pcie-frac 0`
  (or `STRATA_RESIDENT_PIN=0`) gives the mapped mode's exact tokens.
- The engine leaves 4 GB of the RAM it finds free (`STRATA_RESIDENT_HEADROOM_GIB`); when even the experts the GPU does
  not hold do not fit, it says so and runs the plain mapped mode. The server log shows, per request, how many expert
  reads went to the file (`resident RAM: ... blob reads from the file`: 0 in steady use).
- `--low-ram resident|mmap` forces one variant (also on a PC with enough RAM, e.g. to try it).

Time to first token is prompt length / prompt speed: with Q2_0 about 4 s at 4K, 25 s at 32K, under 2 minutes at 128K
and 4.5 minutes at 262K (engine 0.1.13 made long prompts about twice as fast, below).

**Faster prompts (engine 0.1.13):** the prompt is read in chunks of up to 8,192 tokens instead of 2,048 (`--prefill
auto`: the largest chunk whose buffers fit in the expert-cache slots it borrows, and a request borrows only what its
prompt needs); the experts are multiplied by llama.cpp's quantized MMQ kernels instead of being expanded to FP16
first; the next layer's experts stream over PCIe while the current layer's attention runs; the PLE block runs for the
whole chunk at once; unpinned experts are copied by helper threads. Measured on the RTX 5070 12 GB, 64 GB RAM,
32K-token prompt: Q2_0 572 -> 1,290 tokens/s, IQ3_S 383 -> 1,208. Through the server (Q2_0, 128K context): 999 tokens
353 -> 438 tokens/s, 6,927 tokens 529 -> 1,077, 28,584 tokens 584 -> 1,249. Output speed is unchanged. Needles 5/5
(1K-262K). Details and the quality check:
[`bench/results/2026-09-28-prefill-speed`](../bench/results/2026-09-28-prefill-speed/README.md). Existing installs
switch to `--prefill auto` the next time START-HERE / setup.sh starts them. The raw numbers:
[`bench/results/`](../bench/results/). The [paper](paper/Strata-Paper.pdf) explains every number.

**The prompt path borrows expert-cache slots, and that loan is a per-request cost.** `--serve` lays the
prompt path's chunk buffers in the tail of each stage's expert cache, marks the experts there
non-resident for the duration of the read, and streams them back afterwards - so outside a prompt the
cache is whole again for decode. On a 3-GPU machine with ~100 % of the routed mass resident, that
refill (4.95 GiB per stage here) is most of the ~9.5 s per-request floor measured over 1 600 live
requests; it does not scale with the cached context. Three things are therefore done only when they
have to be: the layout is reused when the next segment needs the same chunk and slot range, a loan is
widened in place instead of being handed back and re-taken, and the residency table is re-uploaded to
the devices only when its content actually changed (the batched prompt reads the host table; only the
verify windows read the device copy). `STRATA_PREFILL_STICKY_LOAN=0`, `STRATA_RES_UPLOAD_ALWAYS=1` and
`STRATA_PARK_REFUSALS=0` revert each; `STRATA_PREFILL_LOAN_TIMING=1` prints the loan's cost per lend,
per refill and per request. The decisions are pure and unit-tested in
`include/strata/program/prefill_loan.hpp`; the measurement and the lever ranking are in
[`bench/prefill/`](../bench/prefill/README.md).

## Other GPUs (estimated)

Not measured - estimated from the runs above (same CPU and 64 GB RAM): the GPU part scaled by memory bandwidth, the CPU
part by how many more experts the card's VRAM holds. Treat as **±20%**. Numbers are *prompt / output* tokens/s.
The prompt figures predate engine 0.1.13, which about doubled prompt speed on the measured card; how much of that a
card gains depends on its PCIe link (the experts stream over it), so they are still the older estimates.

| GPU | Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5060 Ti 16GB | Q2_0 | ~341 / ~80 | ~472 / ~87 | ~501 / ~81 | ~492 / ~72 | ~476 / ~62 | ~435 / ~53 |
|  | IQ2_XS | ~291 / ~80 | ~406 / ~77 | ~434 / ~63 | ~426 / ~62 | ~413 / ~51 | ~383 / ~47 |
|  | IQ3_XXS | ~249 / ~66 | ~359 / ~65 | ~381 / ~56 | ~374 / ~54 | ~363 / ~45 | - |
| RTX 3090 24GB | Q2_0 | ~355 / ~128 | ~491 / ~140 | ~521 / ~130 | ~512 / ~115 | ~495 / ~100 | ~453 / ~85 |
|  | IQ2_XS | ~303 / ~131 | ~422 / ~128 | ~451 / ~103 | ~444 / ~102 | ~430 / ~85 | ~398 / ~78 |
|  | IQ3_XXS | ~260 / ~106 | ~374 / ~103 | ~396 / ~89 | ~390 / ~85 | ~378 / ~71 | - |

More VRAM matters more than a faster GPU: every extra GB holds ~700 more experts, and every expert on the GPU is one the
CPU does not have to compute. A 3090's 24 GB takes most of the CPU work away. (Since 0.1.14 the expert profile ranks
all 24,576 experts; before, the cache stopped at 8,000, about 10-14 GB. `tools/make_profile.py` builds a profile from
your own prompts: run the engine once with `--dump-routing trace.bin`, see the tool's help.)

## Which model?

All three are [ISTA-DASLab's GSQ-RCO quantizations](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

| Model | Download | RAM it uses | Speed | Quality |
| --- | ---: | ---: | --- | --- |
| **Q2_0** | 66 GB | ~34 GB experts + ~6 GB | fastest | good |
| **IQ2_XS** | 68 GB | ~36 GB experts + ~6 GB | close to Q2_0 | a bit better |
| **IQ3_XXS** | 76 GB | ~43 GB experts + ~6 GB | slower (more CPU work) | best |

With 64 GB of RAM all three fit (close the browser for IQ3_XXS, and keep its context at 128K or less). With 48 GB only Q2_0 / IQ2_XS may fit. With 32 GB: the Coder (below).

### Or: the Coder (half the experts, for code)

**[Qwen3.8-Flash-Next GSQ-RCO Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)** is
ISTA-DASLab's expert-pruned release: 256 of each layer's 512 experts are kept (still 10 active per token), chosen with
RCO on code, agentic and vision calibration data; its authors report 91.3% of the full model's SWE-bench Verified and
98.7% of LiveCodeBench v6. One size, named IQ1_M for its 1.89 bits per *original* parameter; the kept experts are
stored like IQ3_S (IQ2_S-IQ4_XS gate/up, IQ4_NL/Q2_0 down). Shard 1 is 29.6 GB (experts: 23 GB of RAM), so it runs
on **32 GB of RAM**, and at 262K on 64 GB. Its shard 2 and its vision encoder are the original's files: with the
original installed, setup downloads only shard 1. Strata ships its expert profile (`data/expert-profile-coder.bin`,
the shipped ranking mapped onto the kept experts through the release's `rco-allocation.txt`: 72% of the expert
reads hit the GPU on a 12 GB card). Images work; the experimental speed projection loads and runs on it (it was made
for the full model).

```
START-HERE.bat --setup --family coder
```

### Or: Swift 1.5 (a fine-tune that thinks shorter)

The setup's first question also offers **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)**,
UkisAI's fine-tune of Qwen3.8-Flash-Next, trained to reach the answer with much less thinking (its authors: 63% fewer
thinking tokens, 1.8x sooner answers, under 1% accuracy loss). Same architecture, the same three sizes, its own
vision encoder; Strata runs it at the same speed (4K, IQ2_XS: 465 prompt / 78.7 output tokens/s, vs 467 / 78.3 for
the original). Its authors recommend **IQ2_XS** (their Q2_0 is marked experimental). Its license is the Swift Open
License 1.0 - read it on the model page.

Our small check (8 reasoning questions, default thinking, IQ2_XS): both models got **8/8**; Swift used **1,234**
output tokens in 28 s, the original **2,682** in 46 s - most of the difference from one question the original
thought about for 1,524 tokens. Not a benchmark, but consistent with the claim.

```
START-HERE.bat --setup --family swift --model IQ2_XS
```

## Before you start

You need **only an NVIDIA driver** (version 580 or newer; update it with the NVIDIA App or from
[nvidia.com/drivers](https://www.nvidia.com/drivers)). Everything else is installed for you the first time.

| | |
| --- | --- |
| GPU | NVIDIA **RTX 20, 30, 40 or 50 series**, **12 GB VRAM or more** (8 GB runs, slowly). Measured on an RTX 5070 and an RTX 3090; RTX 20 (Turing, since 0.1.27) was tested by a contributor on an RTX 2070. |
| RAM | **64 GB** recommended (see the table above). |
| CPU | x86-64 with AVX2 (any Intel/AMD desktop CPU from the last ~8 years). AVX-512 (Ryzen 7000/9000) is a bit faster. |
| Disk | ~70-80 GB free for the model, ~6 GB for the MTP layer (+1 GB with images). **Q2_0 on an AVX-512 CPU** also writes a one-time ~40 GB copy of its experts for the fast CPU kernel. An NVMe SSD is strongly recommended. |
| OS | Windows 10/11, or Linux (Ubuntu 22.04/24.04 get everything installed automatically). |

What the first start installs: in this folder `.venv/`, `engine/` and `third_party/`; the model files (`models/`,
`packs/`, `mtp/`, 70-120 GB) in **`Strata-data` next to this folder**, so a new copy of Strata (an update unzipped
elsewhere) finds them and sets itself up the same way. The place is remembered per user (`%APPDATA%\Strata\settings.json`,
`~/.config/strata/settings.json`); `--data-dir` chooses another. Installs from before 0.1.16 are moved there by the next
start (a rename on the same drive; files on another drive are used where they are).
Python 3.12 if you have none (for your user account, no admin), a private Python environment, NVIDIA's CUDA libraries
(from pip, ~0.4 GB), the ready-made Strata engine for RTX 20/30/40/50, the model and the MTP draft layer. If no
ready-made engine fits your PC, it offers to install the build tools (Visual Studio Build Tools + CUDA Toolkit on
Windows, `build-essential` + CUDA on Ubuntu) and compiles the engine for your GPU (asks first; 20-40 minutes once).

---

## Windows

### Double-click `START-HERE.bat`

**The first time** it asks four questions and does the rest:

1. **Which model?** Qwen3.8-Flash-Next (the original) or Swift 1.5 (the fine-tune that thinks shorter).
2. **Which size?** Q2_0, IQ2_XS or IQ3_XXS (it recommends one for your RAM).
3. **How much context?** 8K to 256K tokens (it recommends one for your VRAM).
4. **Images?** yes / no (see [Images](#images-vision)).

Then it downloads and prepares everything (the model is 66-76 GB, so the first start takes a while; an interrupted
download continues where it stopped) and **starts the model**: your browser opens `http://127.0.0.1:8080`, the Strata
app. It has three tabs:
- **Chat:** streaming answers, the model's thinking (folded away once it answers), code with a copy button, pictures when
  images are on, and sampling and thinking-level settings. Chats stay in your browser.
- **Monitor:** what the model is doing (reading the prompt, with progress, or writing, at how many tokens/s); GPU load,
  VRAM, temperature, power and PCIe traffic; CPU, RAM and disk; the context in use; the last requests.
- **About:** the model and engine settings, and the addresses to connect other apps.

`http://127.0.0.1:8080/?q=your question` opens it with a new chat already asking. The API is at
`http://127.0.0.1:8080/v1` for your apps.

**Every time after that**, `START-HERE.bat` just starts the model (30-90 s to load 34-43 GB into RAM). Nothing is
downloaded again. Closing the window stops the model.

```
START-HERE.bat --setup                          install another model, or change context / images
SETUP.bat                                       the same (double-click it)
START-HERE.bat --model IQ2_XS --context 32768 --vision yes --yes     no questions
START-HERE.bat --gguf-dir D:\models\IQ2_XS       use GGUF files you already have
START-HERE.bat --data-dir E:\Strata-data         keep the model files somewhere else
START-HERE.bat --port 8081                      another port
START-HERE.bat --gpu 1                          another GPU (numbered as nvidia-smi; setup picks the one with the most VRAM)
START-HERE.bat --calibrate                      tune the engine for this PC (about 5-10 minutes), then start
```

With more than one model installed, it asks which one to start. `run-<model>.bat` starts a model directly.

**Tuning for your PC (`--calibrate`, engine 0.1.19).** Three engine settings depend on the PC more than on the model:
- the share of the experts missing from VRAM that are copied to the GPU instead of computed by the CPU
  (`--pcie-frac`: a fast PCIe link and a slower CPU want more, a laptop's narrower link less);
- how sure the draft layer must be to add another guess to a check (`--spec-min-p`);
- how many CPU threads compute experts (`--pool-workers`: on CPUs with efficiency cores, fewer can be faster).

The defaults were measured on a Ryzen 5 7600 with an RTX 5070. Setup offers to measure them on your PC after an
install; `START-HERE.bat --calibrate` (Linux: `./setup.sh --calibrate`) does it any time. It measures the output
speed with each setting and keeps one only when it is more than 3% faster. The result is remembered per PC and model
(in the settings file next to the data folder's record), so updates keep it.

### Running it at startup (Task Scheduler)

To have the model up at logon, people start the serve from **Task Scheduler** (or a service). Beware: Windows
throttles such contexts, and the model's ~40 GB expert load then crawls at **~0.05 GiB/s (13-14 minutes)**
instead of **~1.4-1.5 GiB/s (~35 seconds)** - a 24x slower start. Measured on an RTX 5070 Ti + Ryzen 7 9800X3D
+ NVMe, same binary, same args, same cache state:

| How the serve starts | Expert load |
| --- | ---: |
| Double-click / terminal / SSH | 1.42-1.52 GiB/s (~35 s) |
| Task Scheduler with its defaults | 0.05 GiB/s (821-841 s) |
| Task Scheduler with the two settings below | 1.42 GiB/s (35 s) |

In the task's properties set both of these (the defaults are the opposite):

- **Priority level: Normal** (Options tab; the default is Below normal), and
- **Run with highest privileges** (General tab; without it the task runs with a limited user token - which
  also strips `SeLockMemoryPrivilege`, the privilege Windows large pages need).

(Both were changed at once, so the isolated effect of each is not measured.) If the model still starts
slowly, the engine prints a hint under its `loaded ... GiB at ...` line naming this cause.

### Chat in the terminal (optional)

```
.venv\Scripts\python chat.py
```

---

## Linux

```bash
./setup.sh
```

The same questions, the same automatic install (it uses `sudo apt` for Python and, only if it has to compile,
for the build tools), and the same start: `http://127.0.0.1:8080`. Later runs of `./setup.sh` (or `./run-<model>.sh`)
start the model directly. Options as on Windows (`./setup.sh --setup`, `--model Q2_0 --yes`, `--gguf-dir /data/Q2_0`).
Terminal chat: `.venv/bin/python chat.py`.

- **Updating:** `git pull`, then `./setup.sh`: it compiles the engine again when its source changed (a minute or
  two for the changed files). If that compile fails, it says so and starts the engine you had.
- **Other distributions** (Arch, Fedora, ...): install the C++ compiler and the CUDA Toolkit 13 with your package
  manager first (Arch: `sudo pacman -S base-devel cuda`); setup finds `nvcc` on PATH, in `/usr/local/cuda*` and in
  `/opt/cuda*`, and does the rest.
- **WSL** works (Ubuntu 24.04 tested), with one limit: the NVIDIA driver pins only about 1 GB of RAM there, so KV
  streaming (`--kv-resident`) is off and the KV cache stays in VRAM, and the experts are copied to the GPU from
  unpinned RAM (slower prompts than native Linux).

---

## Several Strata servers on one PC (Linux)

You can run more than one model server on the same machine - two models, or the same model twice on two cards.
Each server is its own process with its own weights on its own GPU, and the two things they would otherwise
duplicate are shared instead:

- **The experts are loaded once, not once per server.** The 34-50 GB of expert weights live in one file
  (`--shared-expert-arena`, default `/dev/shm/shared_experts.dat`) that every server maps. The first server to
  start loads them; the others map the same bytes and load nothing, so a second server starts in seconds instead
  of a minute and the machine holds ONE copy of the experts rather than one per server. The file is on
  `/dev/shm` (RAM), so it counts against your RAM exactly once. The engine says which it did:
  `expert arena: SHARED ARENA BORROWED, experts not re-loaded` versus `loaded 46.84 GiB at 1.42 GiB/s`.
  A second server that starts while the first is still loading waits for it rather than reading a half-written
  arena, and if the first one dies mid-load the second takes the load over. Put the file somewhere else with
  `--shared-expert-arena /path/to/file`, or use a private arena per server with `--shared-expert-arena ""`.
  If the shared file cannot be used - Docker's default `/dev/shm` is 64 MiB, the path may be read-only, or the
  file may hold a different model's experts - the engine falls back to a private arena and says why.
  Windows has no shared arena yet (each server keeps its own), and `--mmap-experts` does not use one either.
- **The two servers do not fight over the CPU cores.** The CPU expert path pins its threads to physical cores,
  and two processes that both pinned the same six cores measured **2.5-3x slower each** than two processes on
  different cores - that is what a "one server at 400 % CPU, the other at 80 %" slowdown looks like from the
  outside. Each server now claims cores no other Strata process holds (falling back to the SMT siblings, and
  only then to sharing), and prints what it got:
  `strata generate: cores: host 0 + workers 1-5 (6 physical cores, no other strata process on this machine; lease /dev/shm/strata-core-leases)`.
  With a second server running it takes the other half and says why: `... host 6 + workers 7-11 (6 free physical
  core(s), 1 other strata process(es) on this machine, so this one took a disjoint set ...)`.
  Override with `STRATA_POOL_CORES=6,7,8`, or turn the discovery off with `STRATA_POOL_CORES=none`.

One trade-off on the experts side, and the engine reports it either way: a shared arena is a file, and files are
mapped with 4 KB pages. A PC that has a huge-page pool configured (`vm.nr_hugepages`) would otherwise get 2 MB
pages for its experts, which is better for one server and worse for two. The startup line says which you got -
`MAP_SHARED file /dev/shm/shared_experts.dat` or `hugetlb 2 MB pages` - and `--shared-expert-arena ""` asks for
the private, huge-page arena on a single-server PC.

Both are per-machine, not per-model. Two servers of the **same** model share one arena; two servers of
**different** models cannot, because the file records which model's experts it holds and refuses a mismatch - the
second one falls back to a private arena and says so, which means that model's experts are in RAM twice. If you
run two different models, give each its own file: `--shared-expert-arena /dev/shm/experts-iq2_xs.dat` and
`.../experts-coder.dat`. Serving two requests at the same time from ONE server is a different thing: it is
`--serve-slots N` (opt-in, off by default) - see "Several requests at once" below.

---

## Sharing the GPU with other programs (optional)

By default the model stays loaded until you close Strata. On a PC that also games, renders or runs another model
server, three server options (all off by default; also as keys in `strata-<model>.json`) give the VRAM back:

| Option | Config key | What it does |
| --- | --- | --- |
| `--idle-unload 600` | `"idle_unload_s": 600` | unload the model after 600 s without requests; the next request loads it again |
| `--min-free-vram-mib 11000` | `"min_free_vram_mib": 11000` | load an unloaded model only when that much VRAM is free (it waits up to 15 s for memory being given back), else answer **503** "the GPU is in use by another program" instead of starting into what a game left (with several GPUs it checks the first one) |
| `--before-load "cmd"` | `"before_load": "cmd"` or `["cmd", "arg"]` | a command run before the model is loaded again, e.g. one that unloads another server's model |

`POST /unload` unloads it now (`409` while a request is running) and `POST /load` loads it ahead of a request;
`/health` says `"loaded"`, `/v1/models` lists it as `unloaded` (like llama.cpp's router), `/props` sets
`is_sleeping` and the Monitor shows the state. Unloading ends the engine process - and the image encoder, when images
are on; it is started again first, as at a start - so their VRAM and RAM go straight back. The model files stay in
the OS file cache, so loading again takes seconds while that RAM is not needed elsewhere. Measured on an RTX 5060 Ti
16 GB with Q2_0 in the low-RAM mode: unloading takes ~0.3 s, and a request to an unloaded model answered after
4.6 s (text) or 14.7 s (a picture, image encoder on the CPU).

---

## Using it

The server listens on `http://127.0.0.1:8080` (change with `--port` in setup, or edit the run script).

| API | Endpoint |
| --- | --- |
| OpenAI Chat Completions (stream and non-stream, tools) | `POST /v1/chat/completions` |
| Anthropic Messages (stream and non-stream, tools) | `POST /v1/messages` |
| Model list / health | `GET /v1/models`, `GET /models`, `GET /health` |
| Model properties | `GET /props` (also accepts `?model=<loaded-model-id>`) |
| What the model is doing right now | `GET /status`, `GET /slots` (single slot, busy or idle) |
| Everything the Monitor tab shows (engine, live state, last requests, hardware) | `GET /metrics` |
| The MCP servers, their state and tools ([below](#tools-from-mcp-servers)) | `GET /mcp` |

`/models` and `/v1/models` list only the loaded model, with its context limit and input modalities. `/props` exposes the original chat template, context limit, configured generation defaults (shared settings take precedence), model path and engine version when available. Context means the full engine context, not the resident KV window. `n_predict: -1` means no fixed output cap. Unconfigured sampling fields are omitted. `autoload` has no effect; an unknown `model` returns 404. These metadata endpoints and `/slots` require the API key when one is configured. They do not load, unload or restart models.

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "strata", "messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "max_tokens": 512 }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(model="strata", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

- **Thinking levels: none, low, medium, high.** The model thinks before it answers (streamed as
  `reasoning_content`, Anthropic: `thinking` blocks). Choose how much per request - in the chat page (the "Thinking"
  menu), in `chat.py` (`/think low`), or over the API:

  | API | how |
  | --- | --- |
  | OpenAI | `"reasoning_effort": "none" \| "low" \| "medium" \| "high"` (also `"reasoning": {"effort": ...}`, or `"chat_template_kwargs": {"enable_thinking": false}`) |
  | Anthropic | `"output_config": {"effort": "low" \| "medium" \| "high"}`, `"thinking": {"type": "disabled"}`, or `"thinking": {"type": "enabled", "budget_tokens": N}` (under 2K = low, under 8K = medium, more = high) |

  Without a setting the model uses its own default, **high**. `none` answers at once (fastest); `low` keeps the thinking
  short. The levels are instructions the model was trained with, not a hard token limit: on easy questions all three
  think briefly, on hard ones `high` thinks longest and is most accurate.
- **Streaming.** With `"stream": true` everything arrives as it is made: the thinking, the answer, and tool calls
  (the tool's name first, then its arguments piece by piece, like OpenAI and Anthropic do). While the model reads a
  long prompt the stream sends keep-alives, so agents do not time out; the server window prints progress every
  15 s, and `GET /status` says what it is doing (`reading the prompt`, `answering`, tokens so far). Closing the
  connection or pressing stop in your app really stops the model, so the next request starts at once.
- **Chat apps.** Any app with an "OpenAI-compatible" provider works: base URL `http://127.0.0.1:8080/v1`, any API key.
- **Claude Code** (Strata 0.1.17 or newer): set `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` and
  `ANTHROPIC_MODEL` to a Claude model name it knows (it refuses names it doesn't; Strata ignores the name), plus any
  `ANTHROPIC_AUTH_TOKEN` (or your `api_key`, if you set one).
- **Context.** Chosen in setup (8K-262K). Requests longer than that are refused, never silently cut. A request whose
  `max_tokens` would run past the context is refused too (400); agents that always ask for their full output cap
  can instead get it shortened to the room left: add `"fit_max_tokens": true` to `strata-<model>.json` (or pass
  `--fit-max-tokens` to `serve/server.py`). A prompt that leaves no room at all is still refused.
- **From other devices on your network.** The server listens on your PC only (`127.0.0.1`) unless you say otherwise:
  run setup with `START-HERE.bat --setup --host 0.0.0.0 --api-key some-long-secret` (or add `"host": "0.0.0.0"` and
  `"api_key": "..."` to `strata-<model>.json`). The server window then prints this PC's addresses
  (`from other devices: http://192.168.x.x:8080/`); open that on the other device, or use `.../v1` as an API base URL.
  On Windows the firewall blocks it until you allow it: accept its prompt for Python (private networks), or run
  `New-NetFirewallRule -DisplayName "Strata 8080" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow -Profile Private`
  in an admin PowerShell, and make sure the network is set to Private.
- **From the internet.** Put a tunnel in front of it, for example [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/):
  `cloudflared tunnel --url http://127.0.0.1:8080`. **Set a key first**, or anyone with the link can use your PC:
  add `"api_key": "some-long-secret"` to `strata-<model>.json` (or set the `STRATA_API_KEY` environment variable);
  clients then send it as their API key.

**Conversation cache.** A request that continues a chat reads only the part after what the engine already holds: the
live session, or one of the checkpoints it keeps in RAM (up to 6, ~118 MB each, taken at the start of each new
assistant turn and every 16K prompt tokens). A checkpoint is used only when the prompt starts with exactly its tokens
and pictures. The oldest checkpoint - in practice the end of the system prompt, which every chat of the same client
shares - is kept for good while the rest rotates by least recent use, so a NEW chat that shares that prefix starts
reading after it instead of from token 0. A prompt read from the start is also checkpointed at the end of its system
prompt when that is 2,048 tokens or more (engine 0.1.20; PR #62 + #65), so that root exists for agent clients with long
system prompts and tool lists. Engine options: `--prompt-cache N` (0 = off), `--prompt-cache-every N`,
`--prompt-cache-root N` (0 = no system-prompt checkpoint), `--turn-token ID`.

**Multiple prompts at once.** The engine can park whole prompts in host RAM so a client that switches
between them does not rebuild one from scratch on every switch - an agent that launches subagents with
different system prompts, or a controller and its workers taking turns. `--conversation-cache-mib N` is
the host-RAM budget for that parking (0 turns it off), for example `--conversation-cache-mib 8192`.
Up to **eight** prompts live there at a time (`--conversation-cache-slots N`, default 8). When the
ninth arrives, or when the budget runs out, the engine drops the prompt it has **not used for the
longest**, not the one it parked first: every request that resumes from a parked prompt marks it as
freshly used, so the prompts you actually keep switching between stay warm and the ones you
abandoned are what get pruned. No client session ID is required: only exact token/image prefixes
with matching steering mode are reused. `--prompt-cache 0` also disables parking.
Parking on its own **does not run requests at the same time** - it saves the re-read. Running them
together is `--serve-slots N`, the next section; parking is what makes that possible, because a
request that is not currently in the session has to be saved somewhere.

**Several requests at once (opt-in, off by default).** `--serve-slots N` (2..8) lets
one server process keep N conversations **active at the same time**: while one is reading its prompt,
another is writing its answer, and neither sees the other's tokens. It is off unless you ask for it -
with no flag, or `--serve-slots 0` or `1`, the engine is exactly the serial one it has always been,
byte for byte on the wire.

```
START-HERE.bat --setup ...        then add to the engine args in strata-<model>.json:
  --serve-slots 3 --starve-ms 250 --decode-tokens 10 --conversation-cache-mib 12288
```

How it works, because the trade-off is real: the GPU still does **one thing at a time**. What changed
is that the engine now decides that at the level of a *step* - one verify window, one prompt chunk,
one conversation swap - instead of running a whole request to the end. A conversation that is not
currently in the session is saved to host RAM (the parking cache above) and mounted back when it is
its turn, so `--serve-slots N` needs a parking budget: with `--conversation-cache-mib 0` the engine
says so and stays serial. Requests are cancelled individually (`STOP <id>`), each gets its own
sampling, penalties, drafts and checkpoints, and `GET /slots` shows what the engine thinks each one
is doing.

**How much of an answer one conversation writes before it is swapped out (`--decode-tokens N`).** A
decode step used to be exactly **one verify window**, and a window accepts 1..8 tokens depending on
how many of the MTP's drafts the model agreed with - so a slot emitted ~1-3 tokens and then the
scheduler moved to the next conversation. That is a lot of swapping for very little work: every
possible hand-over boundary pays a save+restore of that conversation's state (hundreds of MB to a
couple of GB). `--decode-tokens N` sets the size of a slot's **turn in tokens**: the engine keeps
running verify windows for the same conversation until it has produced N tokens, and only then lets
the scheduler take the session away. It is a token budget, not a window count, so `--decode-tokens 10`
means "about ten tokens" whatever the acceptance rate happens to be that round; a window may overshoot
it, because a window is the smallest unit of work there is. The request still stops early on EOS, on
`--max-new`, or on `STOP <id>`. `0` (the default) or `1` is the old one-window-per-turn behaviour.
With `--serve-slots 0/1` the flag changes nothing you can see: the serial loop calls the decode step
until the request is done anyway, so it produces the same windows in the same order either way.
Raise it when the swap cost dominates (long conversations, few clients); lower it when you want the
interleaving to feel even. The driver's start-up line prints what is in force
(`decode 10 tokens/turn` vs `decode 1 window/turn`), and it interacts with `--starve-ms`: a longer
turn means a slot holds the session longer before fairness forces a swap.

What it costs: a swap is a save+restore of that conversation's state - hundreds of MB to a couple of
GB - so `--starve-ms` (default 250) is the dial between fairness and memcpy. A slot needs its
conversation to fit the parking budget, and on a PC already holding 47 GB of experts that is the
binding limit, not the GPU. Two conversations do **not** share one decode batch: this is per-request
concurrency, not the packed-batch trick, and the reason is in the paper - the model's recurrent
layers carry one state per sequence. The design, the measurements and the staged plan:
[`docs/STAGE3-CONCURRENCY.md`](STAGE3-CONCURRENCY.md).

**This is new and it has not been measured on a live model yet.** It builds, and its decision logic
is unit-tested without a GPU, but if two clients at once misbehave on your machine, `--serve-slots 0`
turns it off without a rebuild and `STRATA_NO_SWAP=1` turns off only the swapping.

What it costs in RAM is the honest part: a parked prompt is a full snapshot of that conversation -
running state, its checkpoints, the K/V pages it uses and the draft layer's K/V - so it is roughly
what the live session already costs for that chat, again. With a `--layer-split` that means every
stage's share of it, which is the same total: the split divides the state across GPUs, it does not
duplicate it. A checkpoint is ~118 MB, and a snapshot
holds one per assistant turn plus the K/V pages, so a long chat is hundreds of MB to a few GB.
Eight of them is a real number: the budget you set is what decides how many actually fit, and the
engine stops parking before it exceeds it. The engine log prints each snapshot's size
(`snapshot_bytes`) and how many prompts are parked, so you can see what your own prompts cost before
raising the budget. An 8 GiB budget is a cap, not a recommendation for every machine.

**The default budget follows the machine.** 8 GiB is the ceiling, but if the PC has less than that to
spare - after the parking floor below and 4 GiB kept for the next request's own allocations - the
engine starts with what is actually free and says so (`parked-prefix budget 1.2 GiB, not 8.0: ...`).
That is deliberate: a snapshot budget the machine cannot pay for is paid for by dropping the file
cache the experts and the lookup table live in, which makes the server slower than the feature it
just gained. Set `--conversation-cache-mib N` and the engine uses exactly N and does not second-guess
you; `--conversation-cache-mib 0` parks nothing.

**Nothing is reserved per slot.** A common worry is that `--serve-slots 3` at `--max-context 524288`
means three conversations each holding 524288 tokens' worth of K/V, so the budget has to cover the
worst case three times over. It does not. A parked conversation is sized by the tokens it actually
holds, not by what `--max-context` allows it to grow to, and the cache counts what it stored. These
are real parks from one run at `--max-context 524288`:

| conversation | parked snapshot |
|---|---|
| 81 tokens | 226 MiB |
| 150 tokens | 340 MiB |
| 10 800 tokens | 777 MiB |
| 34 000 tokens | 1.4 GiB |
| 95 000 tokens | 2.4 GiB |
| 524 288 tokens (the ceiling) | 7.6 GiB |

The shape of it: a **fixed floor of a couple of hundred MiB** - the recurrent state, the indexer
state and the draft layer's K/V, which every conversation pays whatever its length - plus roughly
**15-30 KiB per context token**, plus about 110 MiB for each checkpoint the conversation retains. So
the budget you need is roughly `slots x floor + sum of your real conversation lengths x the rate`,
not `slots x max_context x the rate`. The startup line prints both figures - the rate
(`a parked conversation costs about N B per context token`) and the worst case
(`a conversation at this --max_context parks in N MiB`) - and `--serve-slots` admission charges each
request against its own prompt length plus the `max_new` it has yet to write, at that rate, instead
of against a flat share of the budget. If a conversation genuinely cannot be parked, the engine says
so before the first request, with both numbers, and it refuses that hand-over instead of destroying
the conversation - see the `park:`, `a conversation at this --max_context` and
`admit slot N refused` lines in the log.

Parking works with a **layer split** (multi-GPU) too: a parked conversation carries the first
stage's running state and K/V, one part for every later stage's own layer range, and the draft
layer's K/V from the stage the drafter lives on. The snapshot is all-or-nothing across the split
as it already was on one GPU - every stage is validated before any of them is written - and the
byte budget counts every part, so a split's snapshots cost what they actually cost. A parked
image is tied to the `--layer-split` it was captured from: change the split and the parked
conversations are refused rather than half-mounted. FP16,
INT8, Q4_0 and identity-layout K8V4 snapshots are supported; the K8V4 draft ring
remains INT8, as in upstream. Windows/HIP runtime coverage must be
reported separately from Linux/CUDA evidence.

Snapshots contain running state, checkpoints, used K/V pages, and draft-layer K/V.
They add host RAM, not another model or VRAM allocation. The byte budget also counts
an incoming snapshot during a switch. After a restore, unchanged K/V pages can be
retained for the next parking operation; growth appends storage without copying
the existing pages. Rewinds refresh the affected pages, and running state and
checkpoints are captured again. Retained active K/V counts against the same byte
budget and is discarded before evicting parked entries under memory pressure.
If reserving space for growth would evict another conversation, parking uses a
full capture instead.
When there is not enough room, the parked conversation that has gone longest
without use is evicted first - not the one parked first - and the eviction count
is logged.
Oversized snapshots or host allocation failures fall back to ordinary prompt processing.
`--conversation-cache-min-free-mib N` (default 2560) additionally requires that
physical-RAM headroom remain available: the engine checks before allocation and
again after capture. Unknown telemetry or insufficient RAM skips parking. Windows
uses `GlobalMemoryStatusEx`, Linux uses `MemAvailable`; these are host-level samples,
not a reservation or enforcement of container/job memory limits.

The shared snapshot core validates all layers and checkpoints - on every stage of a layer split,
and the draft layer - before applying any state. Invalid entries are discarded; transfer/synchronization failure is fatal
rather than permission to continue with partial state. Indexer spare keys and the
moving spare row are preserved, including checkpoint rewinds.
**Mounting and unmounting a conversation (`--serve-slots`, stage 3 S3.1d).** With
`--serve-slots N` (N >= 2) the engine can hand the one live session from one conversation to another
*between steps*, instead of only at a request boundary. There is still exactly one live session and
one set of captured CUDA graphs - a second one does not fit in the VRAM this box has - so a hand-over
is a save and a restore through the same parked-conversation machinery above, never a re-capture.
The order is fixed and it is the part that has to be right:

1. drain any expert-cache swaps still in flight (`apply_pending(true)`) - a snapshot taken over a
   half-applied residency table describes a cache that does not exist;
2. give the prompt loan back (`refill()`) - the loan is the tail of one cache, for the same reason;
3. **validate** the incoming snapshot, which writes nothing anywhere;
4. unmount: save the outgoing conversation into the cache;
5. mount: restore the incoming one into the session;
6. restore the drafter's K/V ring;
7. move the per-slot conversation state in (its live tokens, its checkpoint chain, its steering mode,
   the suffix drafter's history, its sampling and its image positions);
8. re-establish the per-slot device state - the one M-RoPE position table and the control-vector mode.

Validating before saving is what makes a rejected snapshot cost nothing: the outgoing conversation is
still mounted and the request falls back to re-reading, exactly as it does today. A restore that
fails after its first write is still fatal to the session, as it has always been. A conversation the
engine could **not** park (budget or physical-RAM admission refused) is marked un-resumable and
re-read from token 0 rather than mounted back on top of stale K/V - the one case where a silent
"plausible tokens" failure would otherwise be reachable.

**A prompt read that is still in progress is parked too.** A hand-over away from a conversation whose
prompt is only half read used to save nothing and send it back to token 0 on the way in - which is
fine once, and a livelock repeated, because two conversations both reading long prompts would each
throw the other's progress away on every swap and neither would ever finish. A read that has consumed
tokens is now treated like a decode: its state is saved and the read resumes at the token it stopped
at, so `saved`/`restored` are non-zero on those swaps as well. If such a read genuinely cannot be
parked, the engine no longer restarts it forever - after four resets it ends the request with a named
error that says so, rather than spinning invisibly.

**A parked conversation belongs to the request that is still running.** A parked entry is one object -
its branch, its checkpoint chain and its K/V pages - and a new request looks a branch up by *prefix*,
so two chats that share a beginning could collide: the new request matched one of the other slot's
checkpoints, mounted, and thereby removed the entry that held a running conversation's whole branch.
That conversation then could not be handed back and was ended. A parked branch is now claimed by its
owner for as long as that request runs, and another request may not mount it - it reads a little more
of its own prompt instead. The claim is released when the request finishes, so the next request of the
same chat still resumes from the parked prefix exactly as before, and the cache's byte budget still
wins: if everything parked is claimed, the least recently used one is pruned after all.

Every hand-over prints its price on stderr, and the running totals are counted so the cost of
swapping can be measured rather than guessed:

```
strata serve: swap 3: slot 7 -> 9 ok in 142 ms; saved 1073741824 B, restored 805306368 B, parked=3 bytes=2834305024 (total 3 swaps, 401 ms, 3019898880 B out / 2415919104 B in)
```

`STRATA_NO_SWAP=1` turns hand-overs off without a recompile: the wire stays id-tagged, but every
request runs serially against the one mounted slot - 0.1.30's behaviour. `--serve-slots 0` (the
default) and `--serve-slots 1` never take this path at all.

**Running N conversations at once (stage 3 S3.1e-2).** With `--serve-slots N` (N >= 2) and a
conversation cache that has a budget, the serve loop stops being "one request to completion" and
becomes "one step of one slot at a time": the engine reads request lines and admits them (bounded by
N and by physical RAM), asks the registry's `pick()` which conversation runs next, hands the session
over if that is a different one, and runs **one** step - one prompt segment, or one verify window -
before choosing again. A step is the pre-emption point, so a 36 k-token read no longer blocks
everything: its chunks interleave with another conversation's windows. `--starve-ms` (default 250)
is the fairness bound that forces a hand-over; the default is to *stay* on the mounted slot, because
a hand-over is a save+restore of the whole conversation. A decode step is one verify window unless
`--decode-tokens N` makes it a turn of N tokens - see the flag above.

What that buys, and what it does not: the GPU is still serialised (one engine thread issues windows -
the doorbell/pool handshake forbids two), so two conversations share it rather than doubling it. The
win is latency and the fixed cost of prompt reads, not throughput per token.

Three resources are still **one per process**, and the driver treats them as acquired resources
rather than assumptions:

* the **prompt loan** (the tail slots of each stage's expert cache). One owner at a time; a second
  prefill that wants it is *deferred* and served after, and a verify window never runs over a lent
  cache;
* the **M-RoPE position table** (`d_mrope`). In this release an image request (`GENI`) runs alone:
  admission will not bring a second conversation in while one holds non-identity positions, nor an
  image request while anything else runs;
* the **watchdog**. It used to abort the process when the single request stopped beating. It now
  watches every active slot separately: a slot that stalls alone gets `ERR <id>` and its slot is
  destroyed, and the engine aborts only when *every* watched slot stalled - and says which ones.

`--serve-slots >= 2` is **refused at startup** when parking is off (`--prompt-cache 0`,
`--conversation-cache-mib 0`, `--conversation-cache-slots 0`) or when the machine-sized default
budget resolves to 0: a slot switch *is* a save/restore, so without somewhere to park a conversation
"two slots" would mean two conversations overwriting one session. A request that cannot be admitted
gets `ERR <id>` and the process stays up.

The engine log reports parking, restoration, bytes, evictions, individual snapshot
sizes and K/V bytes reused during capture. `STRATA_SNAPSHOT_FULL_CAPTURE=1` disables
retention for diagnostic comparisons. Snapshots are not
persisted across restarts.

**Current limits (v1):** `--serve-slots >= 2` runs several conversations at once inside one process
(above), but the GPU is still serialised - one engine thread, one step at a time - so it is latency
and prompt fixed cost that improve, not tokens per second; an image request runs alone; with the
default `--serve-slots 0` requests are still served strictly one at a time, byte-identically to
0.1.30; one conversation cached live, plus up to eight parked prompts in the cache above,
which prune the one used least recently first - with parking's budget at 0, switching between two chats
re-reads the other one; parking covers a `--layer-split` (multi-GPU) as well as one GPU, but a parked
conversation belongs to the split it was captured from, so changing `--layer-split` drops the parked
ones rather than mounting them into a different layer layout; images only when set up with them (below); no video. **Temperature / top_p / top_k / min_p /
seed** are honored per request (OpenAI and Anthropic fields); with the default adaptive expert tier a sampled result
is not reproducible run to run - for seed-reproducible output add `--adapt-every 100000` (static residency) to the
engine arguments. The run config's optional `sampling` block sets the defaults for requests that leave the fields out
(`"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}`); a request's own fields always win, and with no
block at all a request without sampling keys decodes greedy. The penalties (`presence_penalty`, `frequency_penalty`,
`repetition_penalty`, with `penalty_last_n` capping how many recent tokens they count over, default 64 when any
penalty is set) ride the same path; they count the tokens the request has consumed, so a repetition penalty
suppresses what the model itself just said, not the prompt alone. Since engine 0.1.19 they apply to every token
the speculative decoding checks at once, exactly as if it decoded one token at a time (before, only the first of
each batch got them). That makes requests with penalties 1-11% slower than in 0.1.18: the draft layer guesses
without penalties, so more of its guesses are now rejected. Requests without penalties are unchanged. `top_k` keeps at most 64 candidates: `0` ("off") or anything above 64 uses all 64.

---

## Tools from MCP servers

The chat page can give the model tools from [MCP](https://modelcontextprotocol.io) servers, as LM Studio and Claude
Desktop do: reading your files, fetching web pages, searching, anything an MCP server offers. List the servers in
`strata-<model>.json` under `"mcp_servers"` - the same shape as Claude Desktop's `mcpServers` block, which you can
also paste as it is (key `"mcpServers"`):

```json
"mcp_servers": {
  "files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\me\\Documents\\notes"]},
  "search": {"url": "http://127.0.0.1:3000/mcp", "headers": {"Authorization": "Bearer ..."}}
},
"mcp": {"timeout_s": 60, "max_result_chars": 20000, "max_rounds": 8}
```

Or keep them in their own file and start the server with `--mcp-config path\to\claude_desktop_config.json` (a file
with an `mcpServers` block; add it to the `serve/server.py` line of your run script). Restart Strata after a change.

- **A program** (`command`, `args`, optional `env` and `cwd`) is started by Strata and spoken to over its
  stdin/stdout; `npx`, `uvx`, `python` and friends are found on `PATH` as usual (Node.js is needed for `npx`
  servers). **An address** (`url`, optional `headers`) uses MCP's Streamable HTTP transport (the older SSE-only
  transport is not supported). `"disabled": true` leaves an entry out.
- The servers start with Strata, in the background; the server window says what each one offers
  (`MCP server 'files': 14 tools (...)`), or why it did not start - its tools are then left out and the chat works
  without them. The Monitor tab lists them, and the Sampling drawer has **Use tools from MCP servers** (on by
  default). A server that stops later is started again at its next call.
- In the chat each call shows as a small block (tool, arguments, result); the model reads the result and goes on,
  up to `max_rounds` calls in a row per answer. A tool that fails or takes longer than `timeout_s` (default 60 s)
  gives the model an `error: ...` result instead of ending the chat. Results longer than `max_result_chars`
  (default 20,000 characters) are cut, with a note, before the model reads them. Stop stops a running tool too.
- Only the chat page uses them. API clients (omp, Claude Code, OpenAI and Anthropic SDKs) see the API exactly as
  before and keep their own tools; a request to `/v1/chat/completions` opts in with `"strata_mcp": true` (it then
  gets `strata_mcp` tool events in the stream).

**Security.** MCP tools run on your PC with your user's rights, and **the model decides when to call them** - also
because of what it reads (a web page or a file can contain instructions). Give a filesystem server only the folders
it needs, prefer read-only tools, and don't add servers you don't trust. The tools can only be used from the chat
page itself (a request with another site's Origin or without a JSON content type is refused); if Strata is reachable
from other devices, set an API key.

**Context extension past 262K (rope scaling, EXPERIMENTAL, off unless you pick it).** The model was trained on
262,144 positions (rotary base 1e7). Rope scaling rescales the rotation angles so that longer contexts stay usable,
with llama.cpp's types and flag names. `linear` is Position Interpolation: every angle is shrunk by the factor.
`yarn` keeps the high-frequency angles, interpolates the low-frequency ones, and adds the magnitude correction
that keeps the attention temperature where training put it. **Without the flags nothing changes:** an unscaled
run computes exactly what it did before the feature existed, bit for bit. Scaled contexts need proportionally
more VRAM/RAM for the KV cache and the rope tables (~13 KB and ~0.26 KB per token).

What was measured (contributors' runs, RTX 5080 + IQ3_S, native path, in PR #84): per-position perplexity on the
same tokens, with only the scaling flag changed. At 293K tokens (1.12x the trained length), `yarn` with factor 2
lowered the NLL by 0.18 nats against both `none` and `linear` 2 (2.04 vs 2.22 / 2.22). That is 3-4x the path noise
measured at the same length. `linear` 2 was indistinguishable from `none`. At 2.7K and 32K no arm separated from the
noise. Needle tests do not tell the arms apart: the unscaled model also finds a needle at 413K. Long real-document
Q&A worked with `yarn` 2 at 421K and `yarn` 4 at 714K (8/8 each), and a 1M-token `yarn` 4 run read end to end.
Taken together, use **yarn**. It is still experimental: the numbers come from one machine and one quant.

- setup: `START-HERE.bat --setup --context 393216` asks nothing extra - it picks the method (yarn; one
  question when run interactively) and derives the factor from the final context for you (final context /
  262,144, at least 1: 1.5 at 393K, 2 at 512K, 1 inside the trained range; `--rope-scaling`/`--rope-scale`
  override; an explicit `--rope-scale` is kept as given even when it is too small for the context actually
  served, so check it if you set one). An explicit `--rope-scaling none` for a context past
  262,144 is refused: the setup will not configure a run with the stock angles past the trained range. If
  the RAM check reduces a chosen 384K/512K back inside the trained range, an omitted method adds no
  scaling, and an explicitly chosen one stays at factor 1 - the trained angles, no expansion (not a
  switch for rope as a whole: explicitly supplied rope settings keep their behavior).
- engine: `--rope-scaling none|linear|yarn`, `--rope-scale F`, and the raw ggml knobs `--rope-freq-base`,
  `--rope-freq-scale`, `--yarn-orig-ctx` (default 262,144), `--yarn-ext-factor`, `--yarn-attn-factor`,
  `--yarn-beta-fast` (32), `--yarn-beta-slow` (1). The model file's `rope.scaling.*` keys, when a
  fine-tune ships them, are the defaults the flags override.

The scaling is fixed for the whole run - the engine stores keys in its cache after rotating them, so one
cache must never mix two scalings, and there is no per-request form. Within the trained 262,144 a scaled
run is a slightly different model: the rescaled angles, and `yarn`'s magnitude correction, apply at every
position, not only past the trained end. That is why the setup turns scaling on only for a context past
262,144. Pictures read the same scaled table (their (t, h, w) positions feed it). That should work, but it is
unmeasured: all the runs above are text.

---

## Images (vision)

The model has a vision encoder: [`mmproj-Qwen3.8-Flash-Next-BF16.gguf`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(0.9 GB, a 27-layer ViT plus the projector into the language model). It is **optional**: say yes when the setup asks
"Images?", or run it again with `--vision gpu` (or `--vision cpu`). The setup downloads the encoder, builds a small
helper (`strata-vision`, from llama.cpp's `mtmd` library) and adds it to your start script. Nothing else changes.

| Encoder on | Time per picture | Cost |
| --- | --- | --- |
| **GPU** (recommended) | **0.1-0.5 s** (up to 1,024 image tokens) | ~1.4 GB of VRAM is kept free for it, so the expert cache is smaller: text output is a few % slower (table below) |
| CPU | 10-30 s (pictures are scaled down to ~300 image tokens) | nothing on the GPU |

A picture becomes up to 1,024 tokens of the context (a 640x480 photo: 300). The same picture sent again, as chat apps
do on every turn, is encoded only once.

### Sending a picture

**Terminal chat:** type `/image <path to a picture>`, press Enter, then type your question.

```
you> /image C:\Users\me\Pictures\receipt.jpg
(picture attached: receipt.jpg - now type your question)
you> What is the total on this receipt?
```

**OpenAI API** (an `image_url` part: a `data:` URL, an `http(s)://` URL or a local file path):

```python
import base64
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
img = base64.b64encode(open("photo.jpg", "rb").read()).decode()
r = client.chat.completions.create(model="strata", messages=[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    {"type": "text", "text": "What is in this picture?"}]}])
print(r.choices[0].message.content)
```

**Anthropic API:** an `image` block with a `base64` (or `url`) source, as usual.

JPEG, PNG, BMP, GIF, WebP, TIFF and AVIF work (the last ones are converted to PNG first; agents such as omp send
WebP). Chat apps with image upload work the same way.

### Speed with images on (4K context, measured)

| Model | Prompt tok/s, images off | Prompt tok/s, images on | Output tok/s, images off | Output tok/s, images on |
| --- | ---: | ---: | ---: | ---: |
| **Q2_0** | 541 | 531 | 90.7 | 86.9 |
| **IQ2_XS** | 468 | 458 | 77.1 | 74.0 |
| **IQ3_XXS** | 411 | 401 | 64.7 | 59.5 |

"Off" is the published setup; "on" runs with the encoder loaded on the GPU and its VRAM kept free, so ~1,000 fewer
experts fit in VRAM and the CPU computes a few more per token: 2-8% slower. For text, turning images on changes
nothing else (the output is bit-identical when the VRAM is the same).

A question about a picture (a 640x480 newspaper page = 300 image tokens, a 328-token prompt; the whole request, encoder
on the GPU, measured through the API):

| Model | Answer ("MEN WALK ON MOON") | Picture encoded | Prompt | Output |
| --- | ---: | ---: | ---: | ---: |
| **Q2_0** | 1.9 s | ~0.1 s | 207 tok/s | 65-73 tok/s |
| **IQ2_XS** | 2.3 s | ~0.1 s | 173 tok/s | 50-60 tok/s |
| **IQ3_XXS** | 2.7 s | ~0.1 s | 144 tok/s | 41-50 tok/s |

(Short prompts run below the 4K prompt speed: a 2,048-token chunk is where the prompt path is efficient. Short answers
run below the long-output speed: the first rounds have no draft yet.)

**How it works inside:** the encoder turns the picture into rows of the same width as the model's word embeddings;
Strata puts them where the prompt has `<|image_pad|>` tokens and gives each one its 2-D position (row and column in the
picture; the model uses interleaved M-RoPE). Answers match llama.cpp's multimodal implementation token for token on our
test images.

---

## Experimental speed projection (EXPERIMENTAL, off by default)

**This is an experiment, not a finished feature.** It ships with Strata but stays off unless you turn it on.

A 480 KB control vector for Qwen3.8-Flash-Next (`data/experimental-speed-projection/`, see its README). After each
of layers 4-44 the engine removes one direction from every hyper-connection stream of the residual: `h -= (h . v) v`,
one unit vector `v` per layer, exactly as llama.cpp does with the package's `--cvec-mode project` patches.

**What it changes.** The vector's own package describes it as a **refusal-direction projection**: with it the model
declines far fewer requests (it reports 1 of 50 vs 50 of 50 on its test set), and removing refusals removes a safety
behaviour - you are responsible for what the model writes with it on. It also shifts ordinary answers a little
(measured below). It is not an optimization in the engine: on the same text it costs 0.2-0.4% per token. What a
chat's tokens/s does with it on depends on the text the model writes (length, repetition, how well the drafts land),
so measure it on your own prompts; the Monitor marks every request ESP or stock.

**Turning it on (at setup).** `START-HERE.bat --setup` asks "Turn on the experimental speed projection?" (default:
no), or pass `--experimental-speed-projection on` (`off`, or a path to another vector GGUF). Only for the original
Qwen3.8-Flash-Next, not Swift 1.5. It writes these engine flags (llama.cpp's) into `strata-<model>.json`:

```
--control-vector-scaled <Strata>\data\experimental-speed-projection\Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0
--control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
```

The engine log then says `control vector mode = project, dir = per-layer, layers 4..44 (41 steered)`, and the web
app's About tab lists it. (`--cvec-mode add` is llama.cpp's stock additive mode, for additive vectors.)

**Per request.** A loaded vector is on for every request unless it says otherwise: the web app's Sampling drawer has
a switch, and the API takes `"experimental_speed_projection": false` in the request body (OpenAI and Anthropic; a
config default goes in `"sampling": {"experimental_speed_projection": false}`). Switching drops the conversation
cache once, since the model state was computed the other way. Switched off, the output is token-for-token the stock
model's.

**Measured here** (Q2_0, fixed experts, 2,557 teacher-forced tokens of code, a document and a chat): the top-1 token
changes at 10% of positions, mean KL from the stock model 0.063 nats (max 4.1), perplexity +15% on code, +2.3% on
the document, +0.4% on the chat. Details: `bench/results/2026-09-27-esp/`.

---

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `the NVIDIA driver is too old` | Update the driver (NVIDIA App or nvidia.com/drivers), restart, run `START-HERE.bat` again. |
| Python or the build tools could not be installed | Install what it names (links are printed), then run it again. Everything already done is kept. |
| `port 8080 is already in use` | Strata is already running (look for its window), or another program uses the port: `START-HERE.bat --port 8081`. |
| `cudaHostRegister ... out of memory` in the log | Normal on Windows: the engine pins the experts in per-layer slices instead. Only a problem if the load then fails. |
| `ExpertCache: cudaMalloc(...) failed: out of memory` although VRAM is free | Windows' page file is off or tiny: every allocation on the graphics card is also charged to Windows' commit (RAM + page file). Set the page file to "System managed" (System > About > Advanced system settings > Performance > Advanced > Virtual memory) and restart. Since 0.1.19 the engine retries with a smaller cache instead of stopping, and setup warns about a page file under 4 GB (issue #60). |
| The first start takes minutes | It is reading 34-55 GB into RAM; the second start is faster while the files are in the OS cache. |
| The PC freezes for a few minutes at the start | Normal, most of all the first time (the server window says when it happens): the engine loads the experts into RAM, pins part of it for the GPU and sizes the expert cache. Wait; don't close the window. Still frozen after 10 minutes: restart the PC, close other programs, try again, or pick a smaller size. |
| `the engine stopped unexpectedly (exit code ...)` | The engine process ended mid-answer - usually out of RAM (Linux ends the biggest program: `sudo dmesg \| grep -i -E 'killed process\|out of memory'`). The next request starts it again by itself. If it repeats: close other programs or pick a smaller size. The server also warns at start when the model's experts leave less than ~6 GB of RAM for everything else. |
| Slow output, disk light busy | Not enough free RAM: close other programs, or choose Q2_0 / IQ2_XS. |
| `prompt ... exceeds the context` | The request is longer than the context you chose: run setup again with a bigger `--context`. |
| `the setup refuses --rope-scaling none for a past-trained context` | A context past the trained 262,144 needs the rotary angles rescaled (experimental rope scaling), and the setup will not configure one with the stock angles there. Let it pick (`START-HERE.bat --setup --context 393216` adds yarn and a covering factor), or pass `--rope-scaling linear` or `yarn` yourself. |
| Slower than the tables | The monitor plugged into the GPU and other GPU programs take VRAM from the expert cache; RAM running below its rated speed (enable EXPO/XMP in the BIOS) slows the CPU half. |
| `this server was started without the vision encoder` | The model was set up for text only: run setup again with `--vision gpu`. |
| A picture is refused or `cannot read the image` | The file is not a picture Pillow can open (JPEG, PNG, WebP, GIF, BMP, TIFF, AVIF work). |
| Pictures are slow (10-30 s) | The encoder runs on the CPU: run setup again with `--vision gpu` (needs ~1.4 GB of VRAM). |
| A request never finishes: "reading the prompt", GPU "100%" at low power | The GPU ran out of VRAM (engines before 0.1.9 could end with ~30 MiB free at large contexts). Run `START-HERE.bat` once to get engine 0.1.9 or newer; the log then says `... MiB of VRAM free with everything loaded` (a few hundred) and names the `--vram-reserve-mib` to add if it is low. |
| Generation stops mid-answer, GPU "100%", one CPU core busy | Fixed in engine 0.1.12 (issue #29, a race in the CPU expert pool on big-VRAM cards). Since then a request that stops moving ends with an error instead of hanging (after 2 minutes; 1 minute from 0.1.13): the log says `no progress for ... s ... (issue #29)` with where it stopped, and the next request starts the engine again. If you see that line, please open an issue with it. Engine 0.1.13 adds a stall report under it (what every expert-pool thread and the GPU handshake were doing, memory and page faults) and, on Windows, a `strata-stall-<pid>.dmp` file with every thread's stack: attach both. (`STRATA_WATCHDOG_S` sets the time in seconds; 0 turns it off.) Engine 0.1.14 fixes the stall those reports found (issue #31: with the IQ packs the host could wait forever inside the NVIDIA driver while copying experts in a verify window; the experts are now copied by a GPU kernel, `--pcie-mode dma` restores the old way). |
| `out of memory: cudaFuncSetAttribute` in the log (IQ3_XXS, long prompt) | Fixed in engine 0.1.15: CUDA loaded a kernel's code when it was first needed, and mid-prompt there was no VRAM left for it. Run `START-HERE.bat` (Windows) or `./setup.sh` (Linux) once to update. |
| Anything else | The engine log is `strata-<model>.log` in this folder. |

---

## How it works

<p align="center"><img src="paper/tiers.svg" width="760" alt="memory tiers"></p>

- **GPU (VRAM):** attention and DeltaNet mixers, the gated-residual weights, routers, shared experts, output head, the MTP
  draft layer, the KV cache (from 64K: only its most-read part, the rest streams from RAM), and an **expert cache** that fills the rest of VRAM with the most-used experts (it adapts to
  the conversation while you chat).
- **RAM:** all 24,576 experts, pinned. The CPU computes the experts that are not on the GPU **in place**, at the same time
  as the GPU works on the cached ones (AVX-512 / AVX2 kernels, ggml's for the i-quants).
- **SSD:** the 28.8 GB n-gram table, read a few rows per token through the OS cache.
- **Speculation:** the model's own MTP layer drafts up to 3 tokens; one pass over all 48 layers checks them. 2.4-3.2
  tokens per pass on average. When the reply repeats the context (code edits, quoted text), **prompt lookup** (engine
  0.1.7) drafts up to 5 tokens from the earlier copy, but only where its measured acceptance and cost say it pays:
  code edits 6-11% faster, other text unchanged. The drafts are checked like the MTP's, so the output is the same.
- **Prompts** are processed in 2,048-token chunks with the experts streamed to the GPU over PCIe.

The full story, with measurements, bottlenecks and what comes next: **[docs/paper/Strata-Paper.pdf](paper/Strata-Paper.pdf)**.

---

## Credits and licenses

Strata itself: [MIT](../LICENSE). The model files are not part of it; their licenses apply to them (below).

- Model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team; quantizations:
  [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
  The Coder: [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
  (Apache-2.0 per its card); its support in Strata came from @pjgmobile's PR #54.
  Swift 1.5: [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
  by UkisAI. Their licenses apply to the weights.
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT): the i-quant formats, the GPU dot products and
  dequantizers transcribed in `src/kernels/cuda/iq_kernels.cu`, the CPU backend linked for the i-quant experts, the
  `mtmd` library behind the image encoder (`tools/vision/`), and `gguf-py` used by the tools. See
  `third_party/ggml/LICENSE`.
- Ideas from [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen); references in the paper.
- The web app's font: [Outfit](https://github.com/Outfitio/Outfit-Fonts) (SIL Open Font License 1.1, see
  `serve/web/fonts/OFL.txt`). Its Monitor tab started from @code-martin's dashboard idea (PR #22).
- The experimental speed projection's vector (`data/experimental-speed-projection/`): Qwen Community License 1.0,
  made from the model's activations (see its README).
