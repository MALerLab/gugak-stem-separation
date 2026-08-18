# data/

Symlinks to datasets on server-local NVMe (`~/storage`). No audio lives in the repo — only these pointers. Symlinks are machine-specific and gitignored; recreate them after a fresh checkout.

| Symlink | → target | Dataset |
|---|---|---|
| `gugak_ensemble_71955` | `~/storage/nia-gugak/gugak_ensemble_71955` | AI Hub **71955** — gugak ensemble multitrack (903 songs; `source/`, `labels/`) |
| `gugak_solo_71470` | `~/storage/ngc-gugak` | AI Hub **71470** — solo-phrase clips + MIDI (train-only source pool) |
| `gugak_demo_set` | `~/storage/gugak-demo-set` | **Demo pool** (`v1/raw` · `ingest` · `candidates` · `provenance`) — real-world audio for qualitative demos, incl. YouTube-sourced items; rebuildable from `manifests/demo_sets/` URLs + timestamps |

Each dataset carries its own ground-truth metadata (`metadata.csv`) inside its folder.
