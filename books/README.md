# books/ — local corpus, not committed

The Keshe Foundation books are copyrighted material and are deliberately
**not** part of this repository (`.gitignore` excludes everything here
except the sidecar templates and this note).

To run the books-only pipelines (examples 54–74, `allm benchmark`'s
`books` corpus), place your own copies locally:

```
books/
├── <Book one>.pdf                 # PDFs for figure extraction (optional)
├── book_universal_order.txt       # extracted text sidecars (required)
├── book_structure_of_light.txt
├── book3_keshe_origin_universe.txt
└── sidecar_templates/             # committed: bootstrap templates
```

- Text sidecars (`*.txt`) drive KDP distillation and the benchmark.
- PDFs are only needed for `ALLM_BOOK_IMAGES=1` figure pipelines.
- `ALLM_BOOK_DIR` points the Researcher at a different directory.

Every test that touches a real book guards with a skip when the file is
absent, so the offline suite stays green on a clean checkout.
