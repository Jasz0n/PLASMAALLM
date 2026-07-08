# M7 — Multimodal Research Foundation

Vision source: [`../visuels.md`](../visuels.md)

## Problem

The ecosystem is text-native (`transcript → KDP → graph`). Workshop transcripts reference visuals ("as you've seen in this video") but the Researcher cannot retrieve frames, audio, or synchronized evidence.

## M7 scope (this milestone)

Foundation only — no live video decoding, no student video playback.

| Deliverable | Module | Status |
|-------------|--------|--------|
| Multimodal evidence types | `researcher/multimodal.py` | M7 |
| Video timeline fixture format | `transcripts/Kids/visual/*.json` | M7 |
| Transcript ↔ cue synchronization | `multimodal.sync_transcript_cues()` | M7 |
| `discovery.video` capability | `capabilities/multimodal.py` | M7 |
| `understanding.sync` capability | `capabilities/multimodal.py` | M7 |
| Attach `SyncedEvidence` to packages | extends `KnowledgePackage` | M7 |
| Evidence retrieval for debates | `retrieve_synced_evidence()` | M7 stub |
| Example 34 | offline demo with fixture | M7 |

## Pipeline extension

When `video_fixture_dir` is configured:

```text
… → discovery.workshop → discovery.video → discovery.software
    → understanding.package → understanding.sync → verification.graph → …
```

Capabilities no-op when no fixture directory (same pattern as workshop discovery).

## Fixture format

JSON timeline per workshop — simulates output of a future Video Decoder:

```json
{
  "source_id": "knowledgeSeekerWorkshop9",
  "transcript_ref": "knowledgeSeekerWorkshop9.txt",
  "cues": [
    {
      "timestamp_sec": 732.0,
      "transcript_phrase": "as you've seen in the video",
      "visual": { "description": "...", "frame_start": 2145, "frame_end": 2189 },
      "audio": { "description": "magnet click rhythm" }
    }
  ]
}
```

## Knowledge Package extension

```yaml
KnowledgePackage:
  concepts: [...]
  multimodal_evidence:
    - timestamp_sec: 732
      transcript_excerpt: "..."
      visual: { frame_start: 2145, frame_end: 2189 }
      confidence: 0.87
```

Text `evidence` field is unchanged for backward compatibility.

## Boundaries (unchanged)

- **Researcher** processes raw multimodal data (future: decode video; M7: load fixtures).
- **Teacher/KEL** decides what students learn.
- **Students** receive distilled packages — not raw YouTube streams.

## M8 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Unsynced video curiosity signal | `multimodal.py`, `capabilities/curiosity.py` | M8 |
| Evidence broker for packages | `evidence_broker.py` | M8 |
| Debate evidence resolution | `debate/evidence.py` | M8 |
| Teacher `show_me` API | `teacher/show_me.py` | M8 |
| Example 35 | debate + show me | M8 |

## M9 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| `LoopConfig.enable_debate_evidence` | `loop/learning_loop.py` | M9 |
| `resolve_loop_debate_evidence` | `loop/debate_evidence.py` | M9 |
| `ResearcherLayer.evidence_broker()` | `layer.py` | M9 |
| KEL → Researcher context refresh | `kel_steered_loop.py` | M9 |
| `ALLM_MULTIMODAL=1` in kids KEL run | `kids_kel_run.py` | M9 |
| Example 36 | multimodal learning loop | M9 |

## M10 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| ffprobe/ffmpeg video probe | `video_decoder.py` | M10 |
| Auto-fixture from transcript | `generate_fixture_from_transcript()` | M10 |
| `ensure_workshop_fixtures()` cache | `video_decoder.py` | M10 |
| `auto_generate_video_fixtures` config | `discovery.video` | M10 |
| Dual specialist + debate evidence | `dual_consult_run.py`, example 38 | M10 |
| Example 37 | auto fixture demo | M10 |

## M11 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| `consultation_show_me()` | `teacher/show_me.py` | M11 |
| `show_me_on_reject` on mediated consult | `mediated_consultation.py` | M11 |
| `LoopConfig.enable_consult_show_me` | `learning_loop.py` | M11 |
| Example 39 | consult + show me demo | M11 |

## M12 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| ffmpeg frame extraction | `frame_extractor.py` | M12 |
| Stub vision captioner | `vision_caption.py` | M12 |
| `understanding.vision` capability | `capabilities/vision.py` | M12 |
| `VisualCue.caption` + `frame_path` | `multimodal_types.py` | M12 |
| Example 40 | vision enrichment demo | M12 |

## M13 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Ollama `/api/chat` vision client | `ollama_vision.py` | M13 |
| `OllamaVisionCaptioner` + stub fallback | `vision_caption.py` | M13 |
| `ALLM_VISION_BACKEND=auto/ollama` | env + layer config | M13 |
| Example 41 | Ollama vision demo | M13 |

## M14 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| ffmpeg audio clip extraction | `audio_extractor.py` | M14 |
| Stub + ffmpeg volumedetect analyzer | `audio_analysis.py` | M14 |
| `understanding.audio` capability | `capabilities/audio.py` | M14 |
| `AudioCue.features` + `clip_path` + `analysis` | `multimodal_types.py` | M14 |
| Example 42 | audio enrichment demo | M14 |

## M15 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Stub + tesseract + Ollama OCR | `frame_ocr.py` | M15 |
| `understanding.ocr` capability | `capabilities/ocr.py` | M15 |
| `VisualCue.ocr_text` + `diagram_labels` | `multimodal_types.py` | M15 |
| Example 43 | frame OCR demo | M15 |

## M16 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| LiveKit JWT token bridge | `livekit_tokens.py` | M16 |
| SocialServer active stream client | `social_stream_client.py` | M16 |
| Live observer (stub + RTC SDK) | `livekit_observer.py` | M16 |
| `discovery.livekit` + `understanding.livestream` | `capabilities/livekit.py` | M16 |
| `ResearcherLayer.connect_livekit()` | `layer.py` | M16 |
| Example 44 | LiveKit livestream demo | M16 |

## M17 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| SocialServer ``POST /api/streams/:id/join`` | `SocialServer/streamingService.ts` | M17 |
| SocialServer ``POST /api/streams/:id/leave`` | `SocialServer/streamingService.ts` | M17 |
| `join_live_stream()` client | `social_stream_client.py` | M17 |
| Live audio RTC capture | `livekit_observer.py` | M17 |
| Persistent observer worker | `livekit_worker.py` | M17 |
| Stream → timeline archive | `livekit_archive.py`, `understanding.livekit.archive` | M17 |
| Example 45 | production integration demo | M17 |

## M18 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Full-stack dual loop runner | `examples/dual_consult_run.py` | M18 |
| Researcher + vision + audio + OCR + LiveKit + archive | `ResearcherLayer` via env flags | M18 |
| Debate evidence + consult show-me in one loop | `LearningLoop` + `LoopConfig` | M18 |
| Example 46 | full multimodal stack demo | M18 |

## M19 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Motion / color / diagram analytics | `vision_analytics.py` | M19 |
| `VisualCue.motion_level` + `dominant_colors` + `is_diagram` | `multimodal_types.py` | M19 |
| `understanding.vision.analytics` capability | `capabilities/vision_analytics.py` | M19 |
| `ALLM_VISION_ANALYTICS=1` | env + layer config | M19 |
| Example 47 | vision analytics demo | M19 |

## M20 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Frame sequence sampling | `frame_extractor.py` | M20 |
| Temporal motion tracker | `motion_tracking.py` | M20 |
| `VisualCue.motion_vector` + `motion_score` | `multimodal_types.py` | M20 |
| `understanding.vision.motion` capability | `capabilities/motion_tracking.py` | M20 |
| `ALLM_MOTION_TRACKING=1` | env + layer config | M20 |
| Example 48 | motion tracking demo | M20 |

## M21 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Cross-cue continuity linker | `motion_continuity.py` | M21 |
| `SyncedEvidence.motion_track_id` + `linked_cue_timestamps` | `multimodal_types.py` | M21 |
| `understanding.vision.continuity` capability | `capabilities/motion_continuity.py` | M21 |
| `ALLM_MOTION_CONTINUITY=1` | env + layer config | M21 |
| Example 49 | motion continuity demo | M21 |

## M22 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Cross-workshop identity linker | `object_identity.py` | M22 |
| Workshop 3 magnet fixture | `transcripts/Kids/visual/workshop3_magnet_fields.json` | M22 |
| `SyncedEvidence.object_identity_id` + `linked_source_ids` | `multimodal_types.py` | M22 |
| `understanding.vision.identity` capability | `capabilities/object_identity.py` | M22 |
| `ALLM_OBJECT_IDENTITY=1` | env + layer config | M22 |
| Example 50 | object identity demo | M22 |

## M23 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Teacher visual brief distillation | `visual_distillation.py` | M23 |
| `DistilledVisualBrief` on `KnowledgePackage` | `multimodal_types.py`, `types.py` | M23 |
| `understanding.visual.distill` capability | `capabilities/visual_distillation.py` | M23 |
| Curriculum notes visual briefs for Teacher | `capabilities/curriculum.py` | M23 |
| `ALLM_VISUAL_DISTILL=1` | env + layer config | M23 |
| Example 51 | visual distillation demo | M23 |

## M24 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Teacher visual approval API | `teacher/visual_export.py` | M24 |
| Student-safe export types | `multimodal_types.py` | M24 |
| `understanding.visual.export` capability | `capabilities/visual_export.py` | M24 |
| `KnowledgePackage.student_visual_packages` | `types.py` | M24 |
| `ALLM_VISUAL_EXPORT=1` | env + layer config | M24 |
| Example 52 | student visual export demo | M24 |

## M25 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Visual approval store | `teacher/visual_approval_store.py` | M25 |
| Selective approval workflow | `teacher/visual_export.py` | M25 |
| Student visual delivery | `teacher/student_visual_delivery.py` | M25 |
| KEL loop visual delivery | `loop/learning_loop.py` | M25 |
| `ALLM_VISUAL_DELIVERY=1` | env + LoopConfig | M25 |
| Example 53 | selective approval + delivery demo | M25 |

## M27 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| PDF figure extraction | `book_images.py` | M27 |
| Book → SyncedEvidence | `book_evidence.py` | M27 |
| `understanding.book.images` capability | `capabilities/book_images.py` | M27 |
| Vision/OCR on book frames | `capabilities/vision.py`, `ocr.py` | M27 |
| Example 55 | book visual pipeline demo | M27 |

## M28 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Per-page book brief grouping | `visual_distillation.py` | M28 |
| Provider-scoped brief attachment | `capabilities/visual_distillation.py` | M28 |
| Book curriculum topic on export | `capabilities/visual_export.py` | M28 |
| `source_kind` on briefs | `multimodal_types.py` | M28 |
| Example 56 | book approval + student delivery | M28 |

## M29 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Cross-source concept alignment | `cross_source.py` | M29 |
| `verification.cross_source` capability | `capabilities/cross_source.py` | M29 |
| KEL capstone metrics | `kids_kel_run.py`, example 57 | M29 |

## M30 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Combined-source benchmark report | `evaluation/combined_benchmark.py` | M30 |
| Visual note delivery counting | `student_visual_delivery.py`, `kids_kel_run.py` | M30 |
| Held-out LG ablation (visual delivery) | `examples/58_combined_sources_benchmark.py` | M30 |

## M31 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Teacher visual review service | `teacher/visual_review_service.py` | M31 |
| Teacher review HTTP API + UI | `api/teacher_visual.py` | M31 |
| Workshop + book selective approval demo | `examples/59_teacher_visual_ui.py` | M31 |

## M32 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Teacher approval → export bridge | `teacher/visual_kel_bridge.py` | M32 |
| KEL loop integration | `kids_kel_run.py`, `ALLM_TEACHER_UI_APPROVAL` | M32 |
| Teacher-approved capstone | `examples/60_teacher_approved_kel_loop.py` | M32 |

## M33 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Book corpus audit | `researcher/book_corpus.py` | M33 |
| Export-mode benchmark metrics | `evaluation/combined_benchmark.py` | M33 |
| Auto vs Teacher UI 3-book benchmark | `examples/61_three_book_teacher_benchmark.py` | M33 |

## M34 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Teacher pause/resume session | `teacher/teacher_kel_session.py` | M34 |
| Session API + UI resume | `api/teacher_visual.py` | M34 |
| KEL loop pause integration | `kids_kel_run.py`, `ALLM_TEACHER_UI_PAUSE` | M34 |
| Pause capstone demo | `examples/62_teacher_pause_kel_loop.py` | M34 |

## M35 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Book sidecar bootstrap | `book_corpus.py`, `books/sidecar_templates/` | M35 |
| Full 3-book corpus pipeline | `examples/63_full_book_corpus_pipeline.py` | M35 |

## M36 (this milestone)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Trilogy page verification | `book_corpus.py` `EXPECTED_BOOK_PAGES` | M36 |
| Auto-skip bootstrap when complete | `kids_kel_run.py` | M36 |
| Verified full corpus capstone | `examples/64_verified_full_corpus.py` | M36 |

## M37+ (deferred)

- Full-page extraction benchmark across all 494 book pages

See [`ECOSYSTEM_CAPABILITIES.md`](ECOSYSTEM_CAPABILITIES.md) for capability table.
