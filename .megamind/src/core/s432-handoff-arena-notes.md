# S4.3.2 — the tmpfs handoff arena (LANDED, CPU-tested)

Files (new only, no existing file touched):
`include/strata/core/handoff.hpp`, `src/core/handoff.cpp`, `src/core/handoff_arena_test.cpp`.
CMake: the parent's `EXISTS`-guarded block at `CMakeLists.txt:921-935` activated unchanged
(`strata_handoff` links `strata_core`; `handoff_arena_test` links `strata_handoff` + `strata_engine`).
**The parent's link libraries are correct — verified by configuring with CUDA ON and reading the
generated link line: `libstrata_handoff.a libstrata_engine.a libstrata_kernels.a libstrata_core.a
libstrata_kernels_cpu.a libggml-* cudart librt`.** No CMakeLists.txt edit was needed or made.

## The public API (what S4.3.5 / S4.3.6 call)

Namespace `strata::core`. Everything is host-side: **no CUDA call anywhere in handoff.cpp**.

### start-up sizing (prefill instance, before anything is opened)
```cpp
// pricer(tier_tokens, payload_bytes_out, err) == a wrapper round
// conversation_snapshot_bytes(view_with_exactly_tier_tokens_ids, ss, stages, g, mtp.kv_state(), bytes, err)
bool handoff_size_arena(const std::vector<HandoffTierSpec>& tiers, const HandoffPricer& pricer,
                        HandoffSizing& out, std::string& err);
bool handoff_size_arena(const std::vector<HandoffTierSpec>& tiers,
                        const std::vector<uint64_t>& payload_bytes, HandoffSizing& out, std::string& err);
```
`HandoffSizing{tiers, payload_bytes, slot_bytes, slot_off, slot_total, arena_bytes, max_tokens, describe}`.
`out.describe` is **the line to print** (see "Sizing" below). Tiers must be **ascending** by token count.

### identity
```cpp
struct HandoffSplitStage{int64_t layer_lo, layer_hi;};
struct HandoffSplit{int64_t main_lo, main_hi; std::vector<HandoffSplitStage> stages; bool draft_on_last;};
std::string handoff_split_spec(const HandoffSplit&);                 // filename-safe canonical spelling
uint64_t handoff_geom_hash(const std::array<int64_t,18>& geometry_key, const std::string& split_spec);
std::string handoff_arena_path(const std::string& dir, uint64_t pack_hash,
                               const std::string& split_spec, const std::string& instance);
```
`geometry_key` is the 18-int array `src/core/conversation_state.cpp:14-19` builds — **it is file-local, so
S4.3.5 must pass the same 18 values** (`SavedConversation::geometry` already carries them,
`conversation_cache.hpp:120`). `handoff_split_spec()` is the single canonical spelling; both sides must
use it or `geom_hash` will not agree.

### writer (exactly one process per arena — the prefill instance)
```cpp
HandoffArena a;                       // lease_ms starts at STRATA_HANDOFF_LEASE_MS else 300000
a.open_writer(path, sizing, pack_hash, geom_hash, err);   // creates dir+file; refuses with BOTH numbers
a.set_lease_ms(ms); a.lease_ms();
a.note_client_pid(client_id, pid);    // §3.6 READY-lease immediacy, see deviation 2
a.reclaim(&err) -> std::vector<HandoffReclaim>;           // idle tick; claim() also does it
a.claim(tier, client_id, request_id, HandoffClaim& out, err) -> bool;
a.write_payload(claim, offset, data, bytes, err) -> bool; // no lock held
a.publish(claim, tokens, payload_bytes, payload_hash, err) -> bool;   // THE publish point
a.release(index, nonce, err) -> bool;                     // the ACK path
a.abort_claim(index, nonce, err) / a.cancel(index, nonce, err) -> bool;
a.free_slots(&err) -> "free/total";   a.describe();
```
`HandoffClaim{index, base, payload_off, capacity, tier, nonce}` — `index` is the flat slot id the `CLAIM`
and `DONE` lines name; `tier` is the group **actually used** (may be > requested, §3.8).

### reader (a decode instance; never writes a byte)
```cpp
HandoffArena r;
r.open_reader(path, pack_hash, geom_hash, err);   // O_RDONLY|O_CLOEXEC; checks magic/version/size/hashes
r.read_slot(index, done_line_nonce, my_client_id, HandoffPayload& out, err) -> bool;
HandoffArena::payload_hash_of(out) -> uint64_t;   // the engine's fnv1a64 over the mapped payload
```
`HandoffPayload` RAII-unmaps itself; move-only. Fields: `data, bytes, index, tier, nonce, client_id,
request_id, tokens, payload_bytes, payload_hash, seq`. **The reader must call `payload_hash_of()` and
compare with `payload_hash`** before mounting — that is the last line of defence if the 8-byte publish
store were ever torn.

### free functions
`handoff_pid_alive`, `handoff_proc_start_time` (field 22), `handoff_claim_alive`, `handoff_now_ms`
(CLOCK_MONOTONIC ms).

## The sizing formula (machine-independent)
```
tier_slot_bytes[i] = round_up_64(conversation_snapshot_bytes(tier_max_tokens[i])) + 64 MiB
arena_bytes        = 4096 + Σ_i tier_slot_count[i] * (4096 + tier_slot_bytes[i])
```
Printed the way the engine prints its other start-up numbers (`HandoffSizing::describe`, and
`HandoffArena::describe()` once open):
```
strata prefill: serving on <sock>; slots 2,2,2 (tiers 1024,16384,131072), max 131072 tokens,
  arena 7.99 GiB (8577380352 B), pack hash 0123456789abcdef, geometry hash 5dfa917bae1331c5
```
Measured with the §3.8 default tiers and the measured rate (236 MiB floor + 22 528 B/token):
**7.99 GiB**, i.e. the doc's "≈ 8.2 GiB" — per-tier grouping is what keeps it there.
The `statvfs` refusal line (fatal, no fallback):
```
the handoff arena <path> needs 429765185536 B (400.25 GiB) but /dev/shm/... has 24866926592 B free
  and it needs 429765185536 B; raise --prefill-slots-per-tier, lower --prefill-slot-tiers, or give
  the filesystem more room
```

## Deviations from docs/STAGE4-SPLIT-ROLES.md §3 (also in the header's comment block — ONE truth)
1. **§3.5 step 3 is not implementable as written.** It asks for `payload_bytes`/`payload_hash`/
   `published_ms` "in the same 32-byte `pwrite` as `state`", but with the §3.4 offsets those are at
   104/112/128 while `state` is at 24 — not contiguous. Resolution: one `pwrite` of the contiguous
   **48-byte block [96,144)** (`tokens, payload_bytes, payload_hash, claimed_ms, published_ms, seq`)
   while the slot is still CLAIMED, then `atomic_thread_fence(release)`, then **one 8-byte `pwrite` of
   [24,32) = `state`+`flags` — that store is the publish point.** Invariant preserved; the payload hash
   check catches the pathological torn-store case.
2. **`client_id` is opaque (§4.2 defines it as a u64 the decode instance chooses), so "`client_id`'s pid
   is dead" is not decidable from the slot header.** Added `note_client_pid(client_id, pid)` — the writer
   learns the pid from the HELLO connection / `SO_PEERCRED`. Without a registration the time bound still
   reclaims the slot, so the leak stays bounded either way; the registration only makes it immediate.
   **S4.3.5 must call it at HELLO.**
3. **`geom_hash` is computed by the caller** from the 18-int key + the split-spec string, because
   `geometry_key` is file-local to `conversation_state.cpp` and S4.3.2 may not edit it.
4. **A second `open_writer()` on an arena a LIVE other process created is refused** (one arena, one
   writer). If the recorded creator is dead or its pid was recycled, the arena is **re-initialised from
   scratch** (every slot header → FREE): a restarted prefill instance must not inherit payloads whose
   readers/owners no longer exist.
5. `claimed_ms`/`published_ms` are CLOCK_MONOTONIC ms (comparable across processes on one boot).
6. `payload_hash` is over the slot's **payload region only**, `[slot_base+4096, +payload_bytes)`.
7. Test-only escape hatch: `open_writer(..., bool allow_existing_writer = false)`, default false and
   production must leave it false. It skips ONLY the one-writer refusal, and exists so the CPU test can
   make two processes contend for one slot and prove the claim is exclusive across address spaces.
   It never skips the size/magic/version/hash checks.

## Traps found while implementing (do not reintroduce)
* **`"STRATA-HARENA-V1"` is exactly 16 chars** and does not fit in `char[16]` with a NUL. Compare/copy it
  as 16 raw bytes (`kMagicBytes`), never with `sizeof` on a `const char*` (that is 8, and it silently
  writes a half-magic).
* **`max_tokens()` is the LAST tier, not index 0** — tiers are stored ascending, so index 0 is the
  smallest. (Caught by a test; it would have made `--prefill-max-tokens` and the `toobig` refusal price
  the wrong ceiling.)
* **A refusal message must be built BEFORE `close_()`**, which resets `hdr_`. The one-writer refusal
  printed `pid 0` until it captured the pid first.
* **`reclaim()` and `claim()` must not both take the process mutex**: `claim()` reclaims expired leases
  inside its own locked pass, so it calls `reclaim_locked_()`, which assumes the caller already holds
  **both** `mu_` and the fd lock. Calling the public `reclaim()` from there deadlocks; letting
  `reclaim_locked_()` unlock the fd mid-`claim()` silently drops the lock the claim depends on.
* `flock` is per **open file description**, so two `HandoffArena` objects in one process contend with each
  other — but two *threads* sharing one object share the lock, hence the `mu_` mutex.
* `std::filesystem::exists()`/`stat()` on a path whose parent was just created can race; the writer
  creates the parent with `create_directories` and then decides fresh-vs-existing from `fstat` on the
  opened fd, not from a pre-open `stat` alone.
* In the test, `check(!api(..., err), ("label: " + err).c_str())` is a **bug**: argument evaluation
  order is unspecified, so `err` can be read before the call filled it and the diagnostic prints empty.
  Use `check_err(ok, err, "label")`.

## What was verified (CPU-only; no engine, no model, no GPU, no server)
`handoff_arena_test`: **268 checks, OK**, ~5.2 s, deterministic over 3 runs, leaves nothing behind.
It creates its arenas under `/dev/shm/strata-handoff-test-<pid>/` and removes only that directory.
**`/dev/shm/shared_experts.dat` and `/dev/shm/strata-core-leases` were never opened for writing** —
`/dev/shm` holds exactly those two files before and after.

Coverage: every §3.3/§3.4 offset checked three ways (doc value == header constant == `offsetof`);
split-spec/geom-hash/path identity; the sizing formula recomputed independently + the printed figure;
create + header records; `FREE→CLAIMED→READY→FREE`; **an unpublished (CLAIMED) slot is never readable**;
wrong client / wrong nonce / wrong slot refusals; per-tier claim preference and the big-job refusal
(§3.8 starvation); every ownership refusal (stale nonce cannot publish/cancel/release; a reader handle
cannot claim/write/publish/release; a writer handle cannot read); over-capacity write and publish
refusals with both numbers; **dead-owner take-over** (reclaim + `claim()` itself); **pid-recycled
take-over** via field 22, and `start_time == 0` treated as alive; **leases** — READY-with-no-reader
reclaimed at the bound (the slot-leak fix), CLAIMED reclaimed at the bound, READY reclaimed **immediately**
when the registered reader pid is gone, live reader survives; slot-level identity refusals (a slot whose
own pack/geom hash disagrees, `seq != 1`, `payload_bytes` past capacity, wrong slot magic) — all refused
**before anything is mapped**; arena-level size-vs-header disagreement; a self-inconsistent `HandoffSizing`
refused before any file is touched; version/header_bytes/magic refusals; **`statvfs` refusal on a
too-small filesystem with both numbers and no file left behind**; cross-process via `--role` re-exec —
`flock` contention (exclusive *and* shared-vs-exclusive), **two processes racing for one slot with exactly
one winner** and the loser getting the temporary full-arena reason, a child's published payload read by
the parent and by a second child reader (hash + every byte), wrong-nonce and wrong-client reader children
refused, a live foreign CLAIMED slot not reclaimed by us, and a claim that **waits 1.5 s for a foreign
flock** instead of acting on an unreadable table; a **>1 GiB slot** (1 073 741 825 B) written through the
real `pwrite` path and read back through the mapping with a matching FNV-1a; and a restarted prefill
instance rebuilding a dead creator's slot table.

**Mutation-tested (each neutering must fail the suite):** no READY check → 4; no READY lease → 4; no
dead-owner reclaim → 7; no `statvfs` → 6; **no claim flock → 94**; no publish block → 24; no slot-level
hash check → 2; no `seq` check → 1; no payload-capacity check → 2; no nonce check at publish → 2; no
one-writer rule → 1; no client-dead lease → 1; no arena-size check → 1; no sizing self-consistency → 1;
no ascending-tier rule → 1; **tier cascade inverted to largest-first → 34**.
**Not mutation-testable:** deleting only the `atomic_thread_fence(release)` (keeping both stores) fails
nothing — a fence is a memory-ordering property, not a data property, and no functional test on one box
can see it. The same is true of swapping the two stores' order: the window between them is a few
microseconds and the race test below did not catch it. What IS pinned is that a reader never sees READY
with stale metadata **in practice**: a second thread hammers `read_slot` across 100 publishes of
different payloads (≈2 500 successful reads per run) and every one must carry the round's `tokens`,
`payload_bytes`, `payload_hash` **and** a matching hash over the mapped bytes. The claim zeroes
`tokens`/`payload_hash`, so a reader that caught READY with the previous round's record is caught. The
fence and the store order are there for the same reason `publish_shared_load()` has one; the invariant
they protect is also defended at the end by the reader's own hash check.

## Build note for the parent
`cmake -S . -B /tmp/s432build -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON` **without**
`-DSTRATA_ENABLE_CUDA=ON` leaves `strata_engine` (and therefore the whole `S4.3.x` block) uncreated —
`STRATA_ENABLE_CUDA` defaults to **OFF** (`CMakeLists.txt:35`). The CPU tests need
`-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86` like `build/` has. I configured
`/tmp/s432cudacfg` that way and confirmed `handoff_arena_test` + `strata_handoff` are registered and the
link line is right; I did **not** run the CUDA build (siblings are building, and the scratch build of the
whole engine is not mine to start). The three files were compiled and the test run through a scratch
g++ build with a 8-line `fnv1a64` stand-in copied byte-for-byte from `pinned.cu:594-601` — the only
symbol `handoff.cpp` takes from `strata_core`.

## For S4.3.5 (writer) — the loop the arena expects
`claim()` → `write_payload()` in chunks (no lock, no size limit beyond the slot) → `publish()` → wait for
`ACK` → `release()`. Call `reclaim()` on the idle tick. A `claim()` failure is **`QUEUED`, not `ERR`**
(§3.8/D5). `publish()` failing with "claim is gone" means the slot was reclaimed under you → `ERR slotlost`
and retry. `cancel()` sets the flag; `publish()` then refuses, which is how §4.6 cancellation lands.
Do **not** `take()` the cache entry to write a slot — write from the image in place.

## For S4.3.6 (reader)
`open_reader()` once per connection (it re-checks both hashes and the size). On `DONE <id> <slot> <nonce>
<bytes> <tokens> <ms>`: `read_slot(slot, nonce, my_client_id, payload, err)`, then
`payload_hash_of(payload) == payload.payload_hash`, then hand the bytes to S4.3.1's codec, then `ACK`.
Never write the arena. `read_slot` refusing with "CLAIMED" is not an error — it means the `DONE` line lied.
