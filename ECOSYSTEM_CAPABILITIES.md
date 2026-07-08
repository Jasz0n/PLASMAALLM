# ECOSYSTEM_CAPABILITIES.md — Researcher capability layers

See the vision document: [`../ecosystemCapabilities.md`](../ecosystemCapabilities.md)  
Philosophy and decision cycle: [`RESEARCHER_PHILOSOPHY.md`](RESEARCHER_PHILOSOPHY.md)

## Mapping: vision → code

**Validation legend (M47 audit):**

- **real** — exercised end-to-end on real data with no stand-in backend.
- **auto** — real backend when the tool is reachable (Ollama, ffmpeg,
  tesseract, OpenCV, LiveKit RTC), deterministic stub otherwise; tests
  cover the stub, the listed command exercises the real path.
- **fixture** — validated against fixtures/stubs only; the real backend
  does not exist yet (this is roadmap work, not a hidden gap).

| Layer | Question | Module | Capability name | Validation |
|-------|----------|--------|-----------------|------------|
| Observe | What is uncertain? | `capabilities/curiosity.py` | `observe.curiosity` | real — `examples/33` |
| Analyze | What is missing in the graph? | `capabilities/gap_analysis.py` | `analysis.gap` | real — `examples/33` |
| Missions | What persistent goals are open? | `capabilities/gap_analysis.py`, `missions.py` | `missions.review` | real — `examples/33` |
| L0 Research Planning | What should we acquire next? | `capabilities/planning.py` | `planning.research` | real — `examples/31` |
| L1 Discovery | What is new? | `capabilities/discovery.py` | `discovery.workshop`, `discovery.book`, `discovery.software`, `discovery.repository` | workshop/book real (`examples/25`, `examples/54`); **software fixture** (legacy); repository **real** — reads an actual codebase (`examples/77`, M49) |
| L2 Understanding | What is it about? | `capabilities/understanding.py` | `understanding.package` | real — `examples/25` |
| L3 Verification | How much evidence exists? | `capabilities/verification.py`, `cross_source.py` | `verification.graph`, `verification.cross_source` | real — `ALLM_CROSS_SOURCE_VERIFY=1 examples/57` |
| L4 Curriculum Building | Who needs this? | `capabilities/curriculum.py` | `curriculum.target` | real — `examples/32` |
| L5 Ecosystem Analysis | Who mastered / struggles? | `capabilities/ecosystem.py` | `ecosystem.analyze` | real — `examples/28` |
| L6 Knowledge Economy | Provider reputation? | `capabilities/economy.py` | `economy.ledger` | **fixture** — reputation from citations is a stub (declared non-goal, `RESEARCHER.md`) |
| L7 Self Improvement | What am I bad at? | `capabilities/improvement.py` | `improvement.reflect` | real — `examples/31` |
| L8 Multimodal (M7) | What did we see/hear? | `multimodal.py`, `capabilities/multimodal.py` | `discovery.video`, `understanding.sync` | **fixture** — timeline fixtures stand in for decoded video (`examples/34`, `examples/37`) |
| L8b Vision (M12) | What does the frame show? | `vision_caption.py`, `capabilities/vision.py` | `understanding.vision` | auto — `ALLM_VISION_BACKEND=auto` + `ollama pull llava`, `examples/41` |
| L8c Audio (M14) | What does the sound convey? | `audio_analysis.py`, `capabilities/audio.py` | `understanding.audio` | auto — ffmpeg volumedetect, `examples/42` |
| L8d OCR (M15) | What text is on the diagram? | `frame_ocr.py`, `capabilities/ocr.py` | `understanding.ocr` | auto — tesseract or Ollama vision, `examples/43` |
| L8e LiveKit (M16) | What is happening live? | `livekit_*.py`, `capabilities/livekit.py` | `discovery.livekit`, `understanding.livestream` | auto — RTC SDK + credentials, `examples/45`; stub-validated in CI, **no sustained real-stream run yet** (M51) |
| L8g Vision analytics (M19) | How does the scene move and look? | `vision_analytics.py`, `capabilities/vision_analytics.py` | `understanding.vision.analytics` | auto — OpenCV, `examples/47` |
| L8h Motion tracking (M20) | How does motion evolve over time? | `motion_tracking.py`, `capabilities/motion_tracking.py` | `understanding.vision.motion` | auto — OpenCV frame diff, `examples/48` |
| L8i Motion continuity (M21) | Which cues track the same object thread? | `motion_continuity.py`, `capabilities/motion_continuity.py` | `understanding.vision.continuity` | real — pure logic over cue tracks, `examples/49` |
| L8j Object identity (M22) | Which workshops show the same object? | `object_identity.py`, `capabilities/object_identity.py` | `understanding.vision.identity` | real — pure logic over tracks, `examples/50` |
| L8k Visual distillation (M23) | What should Teacher hand to students? | `visual_distillation.py`, `capabilities/visual_distillation.py` | `understanding.visual.distill` | real — `examples/51` |
| L8l Student visual export (M24) | What subset is student-safe? | `teacher/visual_export.py`, `capabilities/visual_export.py` | `understanding.visual.export` | real — `examples/52` |
| L8m Selective approval (M25) | Which briefs did Teacher approve? | `teacher/visual_approval_store.py`, `teacher/student_visual_delivery.py` | (workflow + loop delivery) | real — `examples/53` |
| L8n Teacher review UI (M31) | HTTP approve/reject + export for workshop/book briefs | `teacher/visual_review_service.py`, `api/teacher_visual.py` | GET `/teacher/` | real — `examples/59` |
| L8o Teacher → KEL bridge (M32) | Selective approval export before loop delivery | `teacher/visual_kel_bridge.py`, `kids_kel_run.py` | `ALLM_TEACHER_UI_APPROVAL=1` | real — `examples/60` (Ollama) |
| L8p Teacher pause/resume (M34) | Interactive review gate before KEL loop | `teacher/teacher_kel_session.py` | `ALLM_TEACHER_UI_PAUSE=1` | real — `examples/62` |
| L8q Book figures (M27) | What diagrams are in the PDFs? | `book_images.py`, `capabilities/book_images.py` | `understanding.book.images` | auto — pypdf + Pillow on real PDFs, `examples/55` |

## Pipeline

[`pipeline.py`](src/allm/researcher/pipeline.py) runs capabilities in order via [`capabilities/registry.py`](src/allm/researcher/capabilities/registry.py). [`ResearcherLayer`](src/allm/researcher/layer.py) is a thin facade.

```
observe → gap → missions → plan → discover (workshop + book + video + livekit) → understand + sync + livestream + vision + audio + ocr + vision.analytics + vision.motion + vision.continuity + vision.identity + visual.distill → verify + visual.export → curriculum → ecosystem → economy → improve
```

## Knowledge tiers

[`knowledge_tier.py`](src/allm/researcher/knowledge_tier.py) classifies each concept as `established`, `emerging`, or `hypothesis` during verification. Tiers flow to `ResearchRecommendation.knowledge_tier`.

## Multimodal evidence (M7)

See [`VISUALS_M7.md`](VISUALS_M7.md) and [`../visuels.md`](../visuels.md). Timeline fixtures in `transcripts/Kids/visual/` stand in for decoded video. `understanding.sync` aligns cues with workshop transcripts and attaches `SyncedEvidence` to packages. `EvidenceBroker` + `teacher_show_me()` + `resolve_debate_evidence()` power debate "show me" retrieval.

## Missions

[`missions.py`](src/allm/researcher/missions.py) stores durable `ResearchMission` goals. Graph gaps auto-open missions; `curriculum.target` attaches `mission_id` when a recommendation topic matches.

## Teacher boundary

Capabilities produce `KnowledgePackage`, `ResearchRecommendation`, and `ResearcherEcosystemMetrics`. Only the Teacher/KEL/planner schedule learning.

## Environment flags

| Flag | Effect |
|------|--------|
| `ALLM_RESEARCHER=1` | Run Researcher before KEL loop |
| `ALLM_RESEARCHER_TARGETING=1` | Per-student recommendation filter in learning loop |
| `ALLM_MULTIMODAL=1` | Enable video timeline fixtures in Researcher (kids KEL run) |
| `ALLM_AUTO_VIDEO_FIXTURES=1` | Auto-generate fixtures from transcript video mentions |
| `ALLM_DEBATE_EVIDENCE=1` | Attach Researcher visuals to unresolved debates in loop |
| `ALLM_CONSULT_SHOW_ME=1` | Teacher show me when mediated consultation is rejected |
| `ALLM_VISION_CAPTIONS=1` | Caption synced visual cues (`understanding.vision`) |
| `ALLM_VISION_BACKEND=auto` | Ollama vision when reachable, else stub (`ollama`, `stub`) |
| `ALLM_VISION_MODEL=llava` | Ollama vision model for frame captioning |
| `ALLM_AUDIO_ANALYSIS=1` | Analyze synced audio cues (`understanding.audio`) |
| `ALLM_AUDIO_BACKEND=auto` | ffmpeg volumedetect when available, else stub (`ffmpeg`, `stub`) |
| `ALLM_FRAME_OCR=1` | OCR on extracted frames (`understanding.ocr`) |
| `ALLM_VISION_ANALYTICS=1` | Motion, color, diagram analytics (`understanding.vision.analytics`) |
| `ALLM_VISION_ANALYTICS_BACKEND=auto` | OpenCV when installed, else stub (`opencv`, `stub`) |
| `ALLM_MOTION_TRACKING=1` | Temporal motion across frame sequences (`understanding.vision.motion`) |
| `ALLM_MOTION_TRACKING_BACKEND=auto` | OpenCV frame-diff when frames exist, else stub (`opencv`, `stub`) |
| `ALLM_MOTION_TRACKING_SAMPLES=3` | Frames sampled per visual cue span |
| `ALLM_MOTION_CONTINUITY=1` | Link cues across workshop timeline (`understanding.vision.continuity`) |
| `ALLM_MOTION_CONTINUITY_MIN_SCORE=0.35` | Minimum continuity score to merge cues into one track |
| `ALLM_OBJECT_IDENTITY=1` | Persist object identity across workshops (`understanding.vision.identity`) |
| `ALLM_OBJECT_IDENTITY_MIN_SCORE=0.30` | Minimum score to merge tracks from different sources |
| `ALLM_VISUAL_DISTILL=1` | Distill visual briefs for Teacher handoff (`understanding.visual.distill`) |
| `ALLM_VISUAL_DISTILL_IMAGES=3` | Max image descriptions per distilled brief |
| `ALLM_VISUAL_DISTILL_QUESTIONS=5` | Max study questions per distilled brief |
| `ALLM_VISUAL_EXPORT=1` | Export Teacher-approved student visual packages (`understanding.visual.export`) |
| `ALLM_VISUAL_EXPORT_AUTO=1` | Auto-approve briefs above confidence threshold (dev/CI) |
| `ALLM_VISUAL_EXPORT_MIN_CONF=0.7` | Minimum brief confidence for auto-approval |
| `ALLM_VISUAL_EXPORT_IMAGES=2` | Max images per student export |
| `ALLM_VISUAL_EXPORT_QUESTIONS=3` | Max questions per student export |
| `ALLM_VISUAL_EXPORT_PERSIST=1` | Persist Teacher/auto approval decisions |
| `ALLM_VISUAL_DELIVERY=1` | Deliver approved visuals into student study memory in loop |
| `ALLM_TEACHER_UI_APPROVAL=1` | Defer export; apply selective Teacher approval then export before KEL loop (M32) |
| `ALLM_TEACHER_MIN_WORKSHOP_CONF=0.75` | Min brief confidence to approve workshop visuals |
| `ALLM_TEACHER_MIN_BOOK_CONF=0.75` | Min brief confidence to approve book visuals |
| `ALLM_TEACHER_UI_PAUSE=1` | Pause KEL loop for interactive Teacher review (M34) |
| `ALLM_TEACHER_PAUSE_TIMEOUT` | Max seconds to wait for Teacher (0 = unlimited) |
| `ALLM_TEACHER_PAUSE_POLL=2` | Poll interval while paused |
| `ALLM_TEACHER_RESUME_FILE` | Override resume flag path |
| `ALLM_BOOK_DIR` | Override Keshe books directory (default: ``PLASMAALLM/books``) |
| `ALLM_BOOK_MAX_FILES=1` | Max PDFs processed per Researcher cycle |
| `ALLM_BOOK_BOOTSTRAP_SIDECARS=1` | Create `.txt` sidecars from templates for corrupt PDFs (M35) |
| `ALLM_BOOK_BOOTSTRAP_SIDECARS=auto` | Bootstrap only when trilogy page verification fails (M36, default) |
| `ALLM_BOOK_SIDECAR_TEMPLATES` | Template dir (default: ``books/sidecar_templates``) |
| `ALLM_BOOK_MAX_PAGES=32` | Max pages extracted per PDF |
| `ALLM_BOOK_PDF_BACKEND=auto` | PDF text backend (`pypdf`, `stub`, `auto`) |
| `ALLM_BOOK_DISCOVERY=1` | Enable book PDF discovery in KEL/loop runners |
| `ALLM_BOOK_IMAGES=1` | Extract PDF figures into visual pipeline |
| `ALLM_BOOK_MAX_IMAGES=24` | Max figures extracted per Researcher cycle |
| `ALLM_CROSS_SOURCE_VERIFY=1` | Align workshop and book concepts after packaging |
| `ALLM_CROSS_SOURCE_MIN_OVERLAP=0.35` | Minimum token overlap for cross-source alignment |
| `ALLM_OCR_BACKEND=auto` | tesseract, else Ollama vision, else stub (`tesseract`, `ollama`, `stub`) |
| `ALLM_OCR_MODEL=llava` | Ollama model for diagram text reading |
| `ALLM_LIVEKIT=1` | Observe live LiveKit workshop streams |
| `ALLM_SOCIAL_API_URL` | SocialServer base URL for ``GET /api/streams/active`` |
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | Same LiveKit credentials as SocialServer |
| `ALLM_LIVEKIT_STREAM_ID` | Explicit stream/room id to observe |
| `ALLM_LIVEKIT_BACKEND=auto` | RTC SDK when installed, else fixture stub (`rtc`, `stub`) |
| `ALLM_LIVEKIT_ARCHIVE=1` | Archive live evidence to timeline fixtures |
| `ALLM_LIVEKIT_WORKER=1` | Buffer live evidence across Researcher cycles |

## Examples

- `examples/31_research_plan_cycle.py` — plan + capability summary
- `examples/32_ecosystem_targeting.py` — per-student recommendation targeting
- `examples/33_researcher_brain_cycle.py` — observe → gap → missions → plan → cycle
- `examples/34_multimodal_sync.py` — video fixture + transcript sync (M7)
- `examples/35_debate_show_me.py` — debate + Teacher show me evidence (M8)
- `examples/36_multimodal_learning_loop.py` — full loop + debate evidence (M9)
- `examples/37_auto_video_fixtures.py` — auto-generate timeline fixtures (M10)
- `examples/38_dual_multimodal_loop.py` — dual specialist + multimodal debate (M10)
- `examples/39_consult_show_me.py` — mediated consultation + show me (M11)
- `examples/40_vision_caption_enrichment.py` — frame captions on synced evidence (M12)
- `examples/41_ollama_vision_caption.py` — Ollama vision model captions (M13)
- `examples/42_audio_enrichment.py` — audio features on synced evidence (M14)
- `examples/43_frame_ocr_enrichment.py` — OCR on workshop frames (M15)
- `examples/44_livekit_livestream.py` — LiveKit live stream observation (M16)
- `examples/45_livekit_production.py` — join API + worker + archive (M17)
- `examples/46_full_multimodal_stack.py` — full stack offline demo (M18)
- `examples/47_vision_analytics.py` — motion, color, diagram analytics (M19)
- `examples/48_motion_tracking.py` — temporal motion tracking (M20)
- `examples/49_motion_continuity.py` — cross-cue motion continuity (M21)
- `examples/50_object_identity.py` — cross-workshop object identity (M22)
- `examples/51_visual_distillation.py` — Teacher visual brief distillation (M23)
- `examples/52_student_visual_export.py` — student-safe visual export (M24)
- `examples/53_selective_visual_approval.py` — selective Teacher approval + KEL delivery (M25)
- `examples/54_book_discovery.py` — Keshe PDF book discovery + KDP packaging (M26)
- `examples/55_book_visual_pipeline.py` — book figures + vision + distillation (M27)
- `examples/56_book_student_visual_delivery.py` — book briefs → Teacher approval → students (M28)
- `examples/57_combined_sources_kel_loop.py` — workshop + book + KEL capstone (M29)
- `examples/58_combined_sources_benchmark.py` — held-out LG benchmark, visual delivery ablation (M30)
- `examples/59_teacher_visual_ui.py` — Teacher HTTP UI for workshop/book brief approval (M31)
- `examples/60_teacher_approved_kel_loop.py` — Teacher UI bridge + KEL visual delivery capstone (M32)
- `examples/61_three_book_teacher_benchmark.py` — 3-book corpus audit + auto vs Teacher UI benchmark (M33)
- `examples/62_teacher_pause_kel_loop.py` — interactive Teacher pause/resume KEL capstone (M34)
- `examples/63_full_book_corpus_pipeline.py` — 3-book audit, sidecar bootstrap, full Researcher (M35)
- `examples/64_verified_full_corpus.py` — verified Keshe trilogy + full Researcher/KEL capstone (M36)

## Non-goals (v0)

Live web crawling, ALLM federation, direct student teaching — see [`RESEARCHER.md`](RESEARCHER.md).
