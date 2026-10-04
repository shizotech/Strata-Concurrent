# S4.3.1 — the SavedConversation payload codec (LANDED, CPU-tested)

Files (new only, nothing existing touched, **no CMakeLists.txt edit**):
`include/strata/core/conversation_wire.hpp`, `src/core/conversation_wire.cpp`,
`src/core/saved_conv_wire_test.cpp`.
The parent's `EXISTS`-guarded block at `CMakeLists.txt:901-916` activated unchanged:
`strata_wire` (static, links `strata_engine`) + `saved_conv_wire_test`.
**The parent's link libraries are correct** — verified by configuring and building for real.

## The public API (what S4.3.2 / S4.3.5 / S4.3.6 call)

Namespace `strata::core`. Everything host-side: **no CUDA call anywhere in conversation_wire.cpp**
(it only takes `fnv1a64` from `pinned.hpp`).

```cpp
// identity the payload carries (deviation 3): pass the SAME two values you put in the slot header
struct WireIdentity { uint64_t geom_hash, pack_hash; };

// ---- writer (S4.3.5) ----
bool conversation_wire_plan(const SavedConversation&, WirePlan& plan, std::string& err);
bool conversation_wire_encode(const SavedConversation&, WireSink&, const WireIdentity&,
                              WireEncoded& out, std::string& err);
// the tier-before-claim decision: plan.total_bytes vs HandoffClaim::capacity
// the publish hash:        enc.payload_hash  == HandoffArena::payload_hash_of()  == §3.4 payload_hash

// ---- reader (S4.3.6) ----
bool conversation_wire_decode(const WireSource&, SavedConversation& image, WireDecoded& info,
                              const WireExpect&, std::string& err);
bool conversation_wire_decode(const uint8_t* data, uint64_t bytes, ...);   // convenience
bool conversation_wire_peek(const uint8_t* data, uint64_t bytes, WireHeaderRecord& hdr, std::string& err);
uint64_t conversation_wire_payload_hash(const uint8_t* data, uint64_t bytes);
std::string conversation_wire_describe(const WireHeaderRecord&);

// sinks / sources
class WireSink { bool write(const uint8_t*, size_t); std::string refusal() const; };
   VectorWireSink, SlotWireSink(int fd, uint64_t payload_off, uint64_t capacity)   // pwrite, no copy
                                          or SlotWireSink(uint8_t* mapping, cap)
class WireSource { uint64_t size(); bool read(uint8_t*, uint64_t off, size_t n);
                   const uint8_t* span(uint64_t off, size_t n); };   // span()==nullptr => codec stages
   SpanWireSource(const uint8_t*, uint64_t), FileWireSource(int fd, uint64_t off, uint64_t n)

struct WireExpect { std::array<int64_t,18> geometry; bool check_geometry;
                    uint64_t geom_hash, pack_hash; bool check_hashes;
                    uint64_t payload_hash; bool check_payload_hash;
                    int64_t stage_parts /*-1 = don't check, else BOTH directions*/;
                    uint64_t max_payload_bytes /*0 = unbounded*/;
                    int max_checkpoint_depth = kWireMaxCheckpointDepth /*8*/; };
struct WirePlan { body_bytes, blob_pool_off, blob_pool_bytes, total_bytes,
                  n_kv, n_checkpoints, n_stage_parts, live_tokens, flags, blobs };
struct WireEncoded { bytes, payload_hash, blob_pool_bytes, blobs };
struct WireDecoded { bytes, payload_hash, geom_hash, pack_hash, geometry[18], layer_lo/hi,
                     n_kv, n_checkpoints, n_stage_parts, live_tokens, flags, cvec,
                     blob_pool_off, blob_pool_bytes, blobs };
```

### The exact call S4.3.5 makes (prefill instance)
```cpp
WirePlan plan; std::string err;
if (!conversation_wire_plan(image, plan, err)) { /* this image cannot be handed off */ }
// price the tier BEFORE claiming: plan.total_bytes must fit tier_slot_bytes
SlotWireSink sink(claim_fd_or_mapping, claim.payload_off, claim.capacity);
WireEncoded enc;
if (!conversation_wire_encode(image, sink, {geom_hash, pack_hash}, enc, err)) { arena.abort_claim(...); }
arena.publish(claim, tokens, enc.bytes, enc.payload_hash, err);   // ONE hash, three places
```
`conversation_wire_plan` allocates nothing and writes nothing — it is the right thing to call when
choosing a tier, and it is cheap (one traversal, no sink).

### The exact call S4.3.6 makes (decode instance)
```cpp
SpanWireSource src(payload.data, payload.payload_bytes);
WireExpect e;
e.geometry = my_geometry_key;  e.check_geometry = true;      // §2.5 step 2
e.geom_hash = my_geom_hash; e.pack_hash = my_pack_hash; e.check_hashes = true;   // §3.10
e.payload_hash = payload.payload_hash; e.check_payload_hash = true;              // §3.4
e.stage_parts = my_conv_stages_count;                       // §2.5 step 2, both directions
e.max_payload_bytes = arena.tier_slot_bytes(payload.tier);  // "declared size exceeds the slot"
SavedConversation image; WireDecoded info;
if (!conversation_wire_decode(src, image, info, e, err)) { /* ERR badpayload, mount NOTHING */ }
conversations.put(std::move(image), /*owner=*/req_id, /*held=*/…);   // THE RECEIVER CLAIMS IT
// then §2.5 step 3 on the engine thread: conversation_snapshot_validate(image, ss, conv_stages, g, mtp.kv_state(), err)
```
`decode` is all-or-nothing: on false the caller's `SavedConversation` is byte-for-byte untouched
(it builds into a local and moves out only on success).

## The format, as built (docs/STAGE4-SPLIT-ROLES.md §2 + the deviations below)

`payload := header_record(256)  body_records  zero_pad  blob_pool`
Header: §2.3's 232-byte field list **pinned by `static_assert`**, padded to 256; the 24 pad bytes carry
`payload_bytes`, `payload_hash`(=0), `reserved`.
Records: `{u32 kind; u32 flags; u64 bytes}` + body; kind 1 KV (fixed 136-byte body), 2 CHECKPOINT,
3 STAGE, and END = `0x21444E45` whose little-endian bytes spell `END!`.
Order: KV array (main layers then the draft) → live → checkpoints → stage_parts → END.
`BLOBREF = {u64 bytes; u64 offset}` from the payload start; zero-length ⇒ `{0,0}` with no pool entry;
every pool entry 64-aligned; gaps and the pool tail are **zero-filled**, which is what makes the payload
deterministic.

## Deviations from §2 (also in the header's comment block — ONE truth)

0. **§2.1's table is wrong.** It says `ConversationCheckpoint` carries "five `ConversationBuffer`s
   (`gdn`, `ple`, `tails`, `dead`, `block_pos`)". It does not — `conversation_cache.hpp:60` declares them
   as plain `std::vector<uint8_t>`. Only `ConversationKv` (`:83`) holds real `ConversationBuffer`s.
   The wire format is identical either way; the codec handles both storage types. **Do not go looking for
   `visit()` on a checkpoint.**
1. **`owner` is not on the wire at all.** §2.4 says the writer always writes `kNoOwner` and the reader
   "must ignore" it — a constant that must never be read is not a field. `encode` never reads
   `image.owner`; `decode` always yields `kNoOwner`. The receiver claims at `put()`. §2.4's rule,
   enforced by construction instead of by convention.
2. **§2.3's 24 padding bytes are used**, not left blank, because the task contract requires a total byte
   count and a content hash inside a payload whose field list is pinned at 232. Same trick as the expert
   arena header's `reserved[]` slots (`pinned.hpp:25-32`). `sizeof == 256` and `kWireHeaderSpecBytes == 232`
   are both asserted, so §2.3's layout is still pinned exactly.
3. **`geom_hash`/`pack_hash` are supplied by the caller** as opaque u64. `geometry_key` is file-local to
   `conversation_state.cpp:14-19` and `handoff_geom_hash()`/`handoff_split_spec()` are S4.3.2's. The codec
   only compares them. **S4.3.5 must pass the same values it wrote into the slot header, and S4.3.6 the
   same values it read from it** — then the payload check and the slot check are one test.
4. **`payload_hash` is zero on the wire and travels out-of-band.** The content hash is
   `fnv1a64(payload[0, payload_bytes))`. If the field held the value the hash would be self-referential,
   and patching it afterwards would make the payload's hash differ from the arena slot's `payload_hash`
   (§3.4), which S4.3.2 computes over the raw slot bytes. Writing zeros makes the two numbers **identical**:
   one hash, checked in both places, produced in one streaming pass, no patch, no second pass.
   A non-zero field on the wire is refused. `conversation_wire_payload_hash()` is exactly
   `fnv1a64(data, bytes, kWireHashSeed)` = what `HandoffArena::payload_hash_of()` returns.
5. **END is a record whose kind IS the ASCII magic** — §2.3 says both "kind 4 END" and "u32 magic
   'E','N','D','!'"; those are the same 4 bytes. Normal 16-byte record header, `bytes == 0`.
6. **Blob alignment is 64 relative to the PAYLOAD start** (§2.3 says "relative to the slot start"). A slot's
   payload begins at `slot_base + 4096` and every `tier_slot_bytes` is `round_up_64(...) + 64 MiB` (§3.8), so
   the two are the same number. Payload-relative is what the codec can guarantee alone.
7. **`live.stage_parts` must be empty; encode refuses otherwise.** The engine already refuses that shape
   ("layer-split parking is not supported", `conversation_state.cpp:432`); the codec says so before any
   byte crosses a process boundary.
8. **Every count is bounded before it sizes anything.** `count * sizeof(element) <= remaining record bytes`,
   plus structural caps (`kWireMaxKvRecords` 4096, `kWireMaxCheckpoints` 1<<20, `kWireMaxStageParts` 64,
   `kWireMaxImages` 1<<16), plus `kWireMaxCheckpointDepth = 8` on the recursive `stage_parts` (the real
   shape is depth 1). A blob reference must land inside the pool, so a corrupt `bytes` cannot make the
   receiver allocate more than the payload itself is. Deleting the count bound makes the test suite die in
   `std::bad_alloc` — that is the bug the bound exists to prevent.

## Traps found while implementing (do not reintroduce)

* **`ConversationCheckpoint` has no `ConversationBuffer` members** (deviation 0). The doc said otherwise and
  the first draft compiled against that assumption and failed.
* **The layout pass must not write.** The encoder runs the same traversal twice (layout, then write). If the
  write pass also assigned offsets, the second pass would append a second set and the offsets would drift —
  symptom: "blob order changed between passes". Two flags, not one: `sink_ != nullptr` (do we write at all)
  and `layout_` (are we assigning or consuming offsets).
* **The record stream is followed by up to 63 zero pad bytes before the pool.** `Cursor` must accept them
  (`c.zeros(pool_begin_ - c.pos())`), and a test that wants to corrupt the END record must **walk** the
  stream to find it — `blob_pool_off - 16` is wrong whenever there is padding, and the corruption then
  silently hits a pad byte instead.
* **The FNV seed must be the offset basis, not 0.** `fnv1a64`'s default seed is
  1469598103934665603; the encoder's running hash started at 0 and the "streaming hash == independent pass"
  check caught it. Now `kWireHashSeed` is a named constant used in all three places.
* **A helper class must write into the caller's error string, not its own.** `Cursor` kept a private
  `std::string error_`, so every truncation/"reserved field is not zero" refusal reached the caller as an
  EMPTY reason. The test that only asserted "some reason was given" passed anyway — which is exactly why
  `check_refusal` alone is not enough; assert the reason's text.
* **Corruption tests must OR the bad bit in, not overwrite the field.** `patch16(p, 6, kWireFlagPartial)`
  also cleared `cvec`/`has_stage_parts`, and the flag-vs-count consistency check then refused for a
  different reason — so the flag check itself was never exercised and deleting it changed nothing.
* **A test that accepts *any* refusal is not a test of that check.** Pin the substring.

## What was verified (CPU-only; no engine, no model, no GPU, no /dev/shm, no server)

`saved_conv_wire_test`: **483 checks, OK, 0.23 s** under ctest in `/tmp/s431build`
(`cmake -G Ninja -S . -B /tmp/s431build -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON
-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && ninja -C /tmp/s431build saved_conv_wire_test`).
The `.sframe` ld lines are pre-existing noise. **The only file the test creates is a `memfd`** (anonymous,
in-memory, `MFD_CLOEXEC`) to exercise the real `pwrite`/`pread` path — nothing on any filesystem.

Coverage: every §2.3 header offset checked three ways (doc value == header constant == `offsetof`, restated
independently in the test); magic spelled both ways; the END kind read as `END!`; round trips for
single-GPU, 2-carve, 3-carve, 4-carve, one-token, **empty conversation**, empty optional buffers, negative
scalars/`cvec` off, and a **>16 MiB buffer spanning `ConversationBuffer` segments** (16 MiB + 12 KB and
20 MiB); `encode → decode → re-encode` byte-identical for all of them; every field of the decoded struct
deep-compared including the recursive checkpoint stage parts; the streaming hash equals an independent pass;
the big payload provably **streamed** (max single write == one 16 MiB segment, never staged whole) and
decoded through a source that never hands over a span; the `pwrite`/`pread` slot path and the mapping path
produce identical bytes and never write outside the slot; `owner` forced to `kNoOwner` on receive.
Refusals: truncation at 4 depths, a slot that cannot serve the bytes its header declares, a chopped pool
(header numbers fixed to match, so the blob references must be what refuses), version ×3, bad magic,
**all 18 geometry slots individually**, geom_hash and pack_hash each naming both values, declared size
past the slot, an absurd declared size, corrupt/negative id counts, a 40-deep `stage_parts` chain (built
with the test-only raised bound, refused at the production bound, and shown to parse when the bound is
raised — so the refusal is the bound, not a broken payload), missing/malformed END, wrong record kinds for
KV/CHECKPOINT/STAGE, record lengths that disagree with their contents, record flags, blob refs outside the
pool / unaligned, `{0,nonzero}`, non-zero reserved/payload_hash, the partial flag, unknown flag bits, an
unaligned pool, one flipped blob byte failing the content hash, the stage-count rule in **both** directions,
a refused decode leaving the caller's image untouched, `live.stage_parts` refused at encode with nothing
written, a sink that runs out of room naming both numbers, and a header that lies about `live_tokens`.

**Mutation-tested (42 mutations; 41 bite).** Deleting any of: the geometry check, the version check, either
hash check, the payload-hash check, the slot-capacity check, the depth bound, the count-vs-room bound
(→ `bad_alloc`), the blobref pool bound, the blobref alignment, the END check, the reserved-zero check, the
partial-flag refusal, the unknown-flag refusal, the stage-count rule, the record-length bound, the
`live.stage_parts` refusal, the ids pad (both directions), the pool zero-fill, the pool tail, the record
body-length agreement (CHECKPOINT and STAGE), the `live_tokens` agreement, the cvec flag, the
has_stage_parts flag, the stage_parts emission, the checkpoints emission, the draft KV entry, the recursive
parts, the imgs, the `used` stamp, the geometry array, the layer range, the zero-length-blob rule, the hash
seed, the KV pad check; or changing pool alignment 64→8; or taking `owner` from the wire.
**Not caught (1):** deleting `Cursor::take`'s own `n > left()` bound while leaving the record-length bound
in place — the outer check refuses first, so the inner one is genuinely redundant defense-in-depth. It is
kept deliberately (a corrupt payload should be stopped at the narrowest place that sees it) and it is not
worth a contrived test.

**Existing CPU tests re-run in the same tree, all green:** `file_expert_source_test` PASS,
`conv_cache_test` PASS, `prefill_loan_test` 128, `serve_swap_test` 173, `serve_proto_test` 140,
`slot_test` 176. `conversation_wire.hpp` is included by nothing except my own two files, so nothing else
could break.

## For S4.3.2 / S4.3.5 / S4.3.6

* The codec's `total_bytes` is the number to price against `tier_slot_bytes` **before** `claim()`. The
  §3.8 64 MiB slack covers the record headers + pool padding (a few KB at most) — it is generous, and the
  plan tells you the truth instead.
* `enc.payload_hash` == `HandoffArena::payload_hash_of(payload)` == the `DONE` line's hash. One number.
* `WireExpect::stage_parts` must be the receiver's **later-stage count** (`conv_stages.count`), i.e. the same
  number `conversation_snapshot_validate` will demand. Pass -1 only if you genuinely do not know it.
* `conversation_wire_peek` is the cheap "is this even a payload from this format" check on a header you have
  already mapped; it does not check geometry/hashes/body.
* A `decode` refusal is `ERR badpayload` (permanent, §4.4) — the same payload fails again. A `plan` refusal
  is a bug in what the prefill instance tried to send, not a transport condition.
