# S3.1b + S3.1c — the slot object and the id-tagged serve protocol

Files (mine):
```
include/strata/program/serve_proto.hpp   NEW  the wire format, one place (NOTE: the task said
                                            src/program/serve_proto.hpp; it lives under include/
                                            because slot.hpp includes it — same API either way)
include/strata/program/slot.hpp          NEW  Slot / SlotRegistry / transitions / pick()
src/program/serve_proto_test.cpp         NEW  CPU-only, incl. the 0.1.30 golden-output test
src/program/slot_test.cpp                NEW  CPU-only
serve/server.py                          slots= handshake, per-id routing, real /slots
serve/test_server.py                     new tests over a PYTHON fake engine (no model)
src/program/generate.cpp                 --serve-slots, READY slots=, INFO slots=, stdin STOP <id>,
                                         request parse -> serve_proto, tag the output lines
CMakeLists.txt                           slot_test + serve_proto_test under STRATA_BUILD_TESTS
```

## Decisions taken (do not re-litigate)

* The optional request id is recognised ONLY when the token after `GEN`/`GENI` and the one after
  that are both bare integers AND a third token exists. `GEN 7 32` therefore stays 0.1.30's
  `max_new=7, ids=[32]` — strictly more compatible than the doc's literal rule.
* `--serve-slots` default **0** = today, byte-identical. `>= 2` turns on the `#<id>` tags and the
  `SLOT` lines. The scheduler itself (S3.1e) is NOT in this change: with slots>=2 the engine is
  still serial but multiplexes the wire.
* `Slot` holds `std::atomic<bool> cancel` so the stdin thread can name a request; the registry is a
  fixed `std::array<Slot, kMaxSlots>` — no allocation in pick(), none on the decode path.
