# Codex Prompt — Build the Autonomous Learning Language Model (ALLM)

You are my lead AI engineer.

Your goal is NOT to build another chatbot.

Your goal is to build an experimental research platform that explores autonomous learning in language models.

The project specification is described in `Plan.md`.

Treat `Plan.md` as the source of truth.

---

## Primary Objective

Build a modular Python codebase where every component can later be replaced independently.

The project should be designed for experimentation, not production.

I want clean architecture, documentation, unit tests, and clear interfaces.

Never create large monolithic files.

---

## Tech Stack

* Python 3.12+
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Accelerate
* FAISS or Chroma for memory
* SQLite or PostgreSQL for metadata
* Pydantic for configuration
* FastAPI (future API)
* Docker
* pytest

---

## Project Structure

Create a professional folder structure.

Example:

```
project/

docs/
Plan.md

configs/

models/

teacher/

students/

planner/

memory/

knowledge/

debate/

exam/

trainer/

collector/

compression/

evaluation/

scripts/

tests/

examples/

experiments/
```

Every folder should have a single responsibility.

---

## Development Rules

Never write placeholder code unless absolutely necessary.

Every module must include:

* documentation
* interfaces
* typing
* tests
* examples

Use SOLID principles.

Keep functions small.

Avoid hidden state.

---

## Phase 1

Build infrastructure only.

Do NOT train a model yet.

Tasks:

* configuration system
* logging
* experiment tracking
* project CLI
* plugin architecture
* dependency injection
* storage layer
* model loading
* dataset loading

---

## Phase 2

Implement the Teacher.

Responsibilities:

* evaluate students
* assign learning goals
* create exams
* measure progress
* maintain global knowledge state

Teacher should expose a clean API.

---

## Phase 3

Implement Students.

Students should:

* load a base language model
* solve tasks
* fine-tune independently
* report confidence
* store failures
* debate with other students

Students should never directly modify the Teacher.

---

## Phase 4

Learning Planner

Implement a planner that computes learning priorities.

Inputs:

* confidence
* exam scores
* novelty
* weakness
* dependencies

Output:

Ordered learning roadmap.

---

## Phase 5

Knowledge Graph

Represent knowledge as concepts.

Each concept should store:

* prerequisites
* related concepts
* confidence
* evidence
* source
* date learned

The graph should evolve over time.

---

## Phase 6

Memory

Implement lifelong memory.

Store:

* successes
* failures
* revisions
* confidence history
* reasoning traces

Never overwrite information without versioning.

---

## Phase 7

Exam Engine

Generate exams automatically.

Include:

* factual questions
* reasoning tasks
* coding tasks
* cross-domain problems

Score every answer.

Store weaknesses.

---

## Phase 8

Debate Engine

Multiple students solve the same problem.

Teacher compares:

* correctness
* reasoning
* confidence
* disagreement

Large disagreements become learning tasks.

---

## Phase 9

Compression Engine

Search for simpler explanations.

Goal:

Replace many disconnected concepts with fewer higher-level principles whenever predictive performance is preserved.

Never lose supporting evidence.

---

## Phase 10

Continuous Learning Loop

Implement the loop from Plan.md.

Measure

↓

Plan

↓

Collect

↓

Learn

↓

Debate

↓

Test

↓

Compress

↓

Update Memory

↓

Repeat

---

## Coding Requirements

Always explain architectural decisions.

Never introduce unnecessary complexity.

Every pull request should improve maintainability.

Prefer composition over inheritance.

Every module should be independently testable.

---

## Research Goal

This project is an experimental AI architecture.

Success is NOT measured by benchmark scores.

Success is measured by whether autonomous curriculum generation and lifelong learning improve the system over time.

If you identify better designs than those described in Plan.md, propose them before implementing them.

Act as both an engineer and a research collaborator.
