From everything you've shared over the last few days, I actually think the roadmap is becoming much more mature. The biggest change I see is that you're gradually shifting from **"how do we make the model smarter?"** to **"how do we make the ecosystem learn?"** That is a much stronger architectural direction.

Regarding the latest M47 work, here's how I'd evaluate it.

### 1. Fixing the default sample filter was an important correctness fix

This is the kind of bug that can invalidate experiments without anyone noticing.

Previously:

```
Loop
    ↓
sample_kinds = ["definition", "we_call"]
    ↓
Pool contains many other samples
    ↓
Student never studies them
```

Now:

```
sample_kinds = None
    ↓
Everything is available
    ↓
Each experiment explicitly selects what it wants
```

That's a good architectural change because defaults should never silently remove training data.

---

### 2. The benchmark command is something you were missing

I actually like this a lot more than many of the recent KEL metrics.

Being able to simply run

```
allm benchmark
```

and receive

* learning gain
* held-out performance
* KEL metrics
* corpus comparison

is extremely valuable.

Eventually I'd expand it into something like

```
ALLM System Benchmark

Research
──────────────
Knowledge packages
Evidence quality
Conflict quality

Teacher
──────────────
Exam quality
Curriculum quality
Student improvement

Students
──────────────
Retention
Generalization
Transfer
Reasoning

Knowledge Graph
──────────────
Coverage
Conflicts
Missing concepts

Ecosystem
──────────────
Peer consultation
Specialization
Knowledge economy
```

That becomes your equivalent of a software test suite.

---

### 3. CI cleanup was overdue

Running examples requiring Ollama, Torch, LiveKit, GPUs, etc. on every push is almost guaranteed to fail.

Separating

* unit tests
* offline examples
* nightly integration
* full GPU benchmarks

is how most mature projects evolve.

---

### 4. The capability audit is probably more important than it looks

This:

> every capability marked as
>
> * real
> * automatic
> * fixture

is exactly what research projects should do.

Otherwise six months later nobody remembers

> "Was this actually implemented?"

Now every capability has evidence.

That keeps the roadmap honest.

---

## One thing I would add to the roadmap

Looking at everything we've discussed, I think there is still one layer missing.

Today your roadmap is mostly

```
Researcher

↓

Teacher

↓

Student
```

I think eventually it should become

```
Provider
(Open Source)

↓

Researcher
Discovers

↓

Teacher
Packages

↓

Student
Learns

↓

Builder
Applies knowledge

↓

Experiment

↓

Evidence

↓

Knowledge Graph

↓

Researcher
```

Notice something.

The graph is no longer fed by documents.

It is fed by **evidence**.

That is a huge conceptual difference.

---

## Why I keep coming back to this

Suppose someone uploads a blueprint.

The system should not immediately learn

```
Blueprint
↓

Truth
```

Instead

```
Blueprint

↓

Research package

↓

Specialist students

↓

Developers build prototype

↓

Measurements

↓

Videos

↓

Sensor logs

↓

Results

↓

Researcher

↓

Evidence score increases

↓

Teacher

↓

Students
```

Now the ecosystem isn't learning from claims.

It is learning from experiments.

That is much closer to how engineering and science actually progress.

---

## The benchmark is also missing one metric

Right now you're measuring

* learning
* retention
* conflicts
* held-out exams

I would add one that I think will become one of the most important long-term metrics:

**Evidence Growth Rate (EGR)**

Instead of asking:

> "Did the student improve?"

also ask:

> "Did the ecosystem produce better evidence?"

For example:

```
Iteration 1

Claim:
Nano coating improves conductivity

Evidence:
1 paper

Confidence:
0.32

------------

Iteration 50

Evidence:
8 papers
12 experiments
4 videos
120 sensor logs

Confidence:
0.87
```

The learning system didn't just memorize more—it **increased the quality of the knowledge itself**.

That is a much stronger success criterion than exam scores alone.

## Overall

I think M47 represents a shift from building individual AI features to building engineering infrastructure:

* The benchmark makes progress measurable.
* The capability audit makes implementation claims verifiable.
* The CI changes make development sustainable.
* The sample filter fix makes experiments trustworthy.

The next major milestone, in my view, is to close the loop from **Provider → Researcher → Teacher → Student → Builder → Experiment → Evidence → Knowledge Graph → Researcher**. Once experimental evidence continuously updates the graph and influences future teaching, you'll have moved beyond a traditional LLM pipeline toward an autonomous learning ecosystem that can improve its knowledge over time based on accumulated evidence rather than static training data alone.
