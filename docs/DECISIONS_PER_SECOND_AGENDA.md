# Decisions per Second: A Research Agenda for Fast, Grounded AI Systems

This document proposes a research agenda for evaluating fast AI systems by verified decision throughput rather than raw token throughput.

It is a future-facing design note, not a claim that this repo already implements real verifier-backed decision engines.

## 1. Motivation

Tokens/second measures generation throughput. It does not directly measure whether a system makes correct, grounded, and verifiable choices.

The next evaluation target is:

`D/s = verified decision utility per second`

The main question is when higher TPS actually converts into higher decision quality, and when it only produces mistakes faster.

## 2. Defining a Decision

A decision is a committed action proposal with:

- an objective
- a verifier
- a correctness criterion
- a confidence estimate
- a latency cost

Let a decision `d` have:

- input state `x`
- action `a`
- confidence `c`
- verifier `V`
- outcome `y = V(x, a)`

A decision is correct when it satisfies domain-specific acceptance criteria.

### 2.1 Coding agent

Decision:
- a code change or patch candidate

Correctness:
- passes required tests
- preserves buildability or type correctness
- satisfies task constraints

Verifier:
- test suite
- compiler or type checker
- static analysis
- optional human review for underspecified tasks

### 2.2 Financial signal

Decision:
- a trade recommendation with entry, exit, and risk policy

Correctness:
- positive realized or expected utility under evaluation horizon
- bounded downside within declared risk envelope
- acceptable confidence calibration

Verifier:
- backtest engine
- delayed market outcome
- risk model
- compliance checks

### 2.3 Medical triage

Decision:
- a patient routing action such as home care, urgent consult, ED, or ICU

Correctness:
- clinically appropriate routing
- supported by cited evidence
- no contraindication or safety violation

Verifier:
- clinician gold label
- triage guideline engine
- evidence checker
- retrospective outcome review

## 3. Decisions/Second Metric

Let decision `i` have:

- difficulty weight `w_i`
- correctness value `z_i in {-1, 0, 1}`
- confidence `c_i`
- confidence penalty `p_i`

Define:

`S_i = w_i * z_i - lambda * p_i`

Then:

`D/s = (sum_i S_i) / T`

Where `T` is wall-clock time.

### 3.1 Difficulty normalization

Easy decisions should count less.

Possible weighting sources:

- baseline solve rate
- branching factor
- tool depth
- ambiguity score
- domain-specific rarity

### 3.2 False-confidence penalty

Wrong high-confidence decisions should be punished harder.

One simple choice:

- `p_i = c_i^alpha` if wrong
- `p_i = 0` otherwise

### 3.3 Hardware-agnostic requirement

D/s should depend on:

- correctness
- calibration
- wall-clock latency
- task difficulty

It should not directly depend on:

- GPU type
- quantization format
- node count

Hardware matters only through achieved latency and throughput.

## 4. Relationship Between TPS and D/s

Higher TPS improves D/s only when extra tokens are converted into useful verified search.

### 4.1 When TPS helps

Higher TPS raises D/s when it enables:

- more candidate branches
- faster tool and verification cycles
- more retries under fixed latency
- wider search with bounded verifier cost

### 4.2 When TPS does not help

Higher TPS does not raise D/s when:

- verification is the bottleneck
- tool latency dominates
- branch quality collapses
- confidence is miscalibrated
- correlated errors accelerate

### 4.3 Comparison table

| Scenario | TPS | Verification quality | Tool latency | Expected D/s |
|---|---:|---|---|---|
| Fast generator, weak verifier | High | Low | Medium | Low to medium |
| Fast generator, strong verifier, cheap tools | High | High | Low | High |
| Slow generator, strong verifier | Low | High | Low | Medium |
| High TPS, high correlated error | High | Medium | Low | Low or negative |

## 5. Tight Feedback Loop Architecture

A latency-optimized Decision Engine should fuse:

- fast generation
- tool use
- symbolic verification
- branch selection
- commit or reject logic

### 5.1 Pipeline

1. `Draft`
- generate candidate actions or branches

2. `Tool`
- code execution, DB lookup, retrieval, calculator, market data, guideline retrieval

3. `Verifier`
- tests, rules engine, theorem checker, citation checker, risk model

4. `Merger`
- compare branches, score candidates, commit best verified option

5. `Feedback`
- rejected branches become repair prompts or pruning signals

### 5.2 Reference latency budget

Example coding budget:

- generation: `50-150 ms`
- tool execution/tests: `100-500 ms`
- symbolic verification: `20-150 ms`
- branch selection/merge: `10-50 ms`
- final formatting/citation: `10-30 ms`

The exact split varies by domain, but the main point remains: token generation is only one part of the decision loop.

## 6. Failure Modes at High Speed

### 6.1 Cascading false positives

Problem:
- many weakly verified branches get committed rapidly

Detection:
- rising verifier disagreement
- drop in branch acceptance precision
- spike in high-confidence failures

Circuit breaker:
- raise verifier threshold
- reduce branch fanout
- switch to abstain-first mode

### 6.2 Tool amplification

Problem:
- system floods tools or APIs with speculative calls

Detection:
- tool-call rate exceeds budget
- queue latency spikes
- redundant call similarity rises

Circuit breaker:
- cap tool concurrency
- deduplicate requests
- force cache or retrieval reuse

### 6.3 Correlated fast error

Problem:
- many branches share the same wrong assumption

Detection:
- low branch diversity
- repeated verifier failure signature
- repeated rejection at same logical step

Circuit breaker:
- inject diversity prompting
- require contradictory branches
- pause commit until independent evidence appears

### 6.4 Confidence runaway

Problem:
- internally consistent but externally wrong branches produce inflated confidence

Detection:
- confidence-verifier mismatch
- calibration drift
- rising confidence with falling external acceptance

Circuit breaker:
- confidence clipping
- recalibration layer
- mandatory external verification for high-impact actions

## 7. Research Gaps

### 7.1 Branch-level verification

Problem:
- efficient verifiers for partial reasoning branches remain weak

Experiment:
- compare final-answer verification vs branch-prefix verification

Dataset:
- SWE-bench
- LiveCodeBench
- MATH

Success criterion:
- higher D/s at equal error rate through earlier rejection and repair

### 7.2 Confidence calibration under search

Problem:
- Best-of-N and tree search distort confidence semantics

Experiment:
- measure calibration before and after branch search and merge

Dataset:
- ARC
- MATH
- finance backtest slices
- clinical triage sets

Success criterion:
- lower false-confidence penalty at same throughput

### 7.3 Tool-aware latency allocation

Problem:
- high TPS does not help if tool and verifier budgets are misallocated

Experiment:
- adaptive latency scheduler vs fixed scheduler

Dataset:
- SWE-bench with tests
- retrieval-heavy QA
- simulated triage with citation lookup

Success criterion:
- improved D/s under fixed wall-clock budget

## 8. 12-Month Timeline

| Quarter | Focus | Deliverable |
|---|---|---|
| Q1 | Formalization | decision taxonomy, D/s metric, confidence penalty definitions |
| Q2 | Benchmark substrate | multi-domain benchmark harness for coding, finance, and triage |
| Q3 | Decision Engine prototype | generation + tool + verifier + branch manager runtime |
| Q4 | Research studies | branch verification, calibration-under-search, latency allocation |

### Q1

- define decision schemas
- define difficulty weights
- define verifier interfaces
- publish metric spec

### Q2

- build benchmark runners
- implement wall-clock accounting
- collect baseline TPS vs D/s curves

### Q3

- implement branch engine prototype
- integrate tool loop and symbolic verifier loop
- add circuit breakers and logging

### Q4

- run ablations
- compare raw TPS scaling, search scaling, verifier scaling, and tool-latency scaling
- produce publication-ready D/s benchmark

## 9. Summary

The key distinction is simple:

- TPS measures how fast the model can speak
- D/s measures how fast the system can make correct, grounded, verifiable choices

Higher TPS matters only when it is converted into:

- more verified search
- tighter tool-feedback loops
- better branch management
- stronger calibration

Fast future systems should therefore be evaluated not only as generators, but as decision engines.
