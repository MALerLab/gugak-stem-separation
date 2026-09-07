# demo_web — ISMIR 2026 LBD listening page

Static, data-driven GitHub Pages site. Nothing about songs is hard-coded: the song list lives
in `manifest.yaml`, the build knobs in `configs/demo_web.yaml`, and `site/index.html` renders
whatever `site/data.json` says.

| file | role |
| --- | --- |
| `manifest.yaml` | **the one file to edit** when the song list changes (three sections, page order) |
| `configs/demo_web.yaml` | paths to renders / eval rows, MP3 knob, audibility gate, class order, publish target |
| `scripts/build_demo_web.py` | cuts windows, gates stems, encodes MP3 + peaks, writes `site/data.json` + `site/README.md` |
| `scripts/publish_demo_web.sh` | rsyncs `site/` into the `jae-gye/gugak-demo` clone and pushes (Pages root) |
| `site/index.html` | the page (wavesurfer.js from CDN, no build step) — the only tracked file under `site/` |

```bash
uv run python scripts/build_demo_web.py        # idempotent; --force re-encodes every MP3
bash scripts/publish_demo_web.sh               # commit + push the built site
```

Audio, peaks and `data.json` are gitignored here and live only in the Pages repo.
