# Cleaned workshop transcripts (training source)

Run `examples/13_kids_transcripts_kdp.py` to regenerate from raw ASR files.

| Path | Use |
|------|-----|
| `*.txt` (this folder) | **Full cleaned workshop** — all speakers, every turn kept except AV/logistics noise |
| `mk/*.txt` | **Mr Keshe only** — same cleaning, no summarisation; best for teacher-focused training |
| `../digest/*.md` | Optional LLM index — **lossy**, do not use as training data |

Raw originals in `../` are never modified.
