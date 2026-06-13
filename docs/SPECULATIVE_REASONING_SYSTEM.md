# Speculative Reasoning and Real-Time Multi-Agent Orchestration

This note sketches a future-facing systems design for exploiting very high-throughput GPU-resident inference loops in branch-heavy reasoning workloads.

It is intentionally a design proposal, not an implemented feature of this repo today.

## 1. Architecture

### 1.1 Objective

Assume a serving substrate where an individual verifier-class stream can sustain roughly 1000+ tokens/sec on a large model, while lightweight draft components generate candidate continuations or reasoning branches even faster.

The goal is not merely to run many chat agents in parallel. The goal is to maintain a bounded, verifier-backed branch graph whose nodes represent partial reasoning states, code patches, or candidate decisions.

### 1.2 Task partitioning across agents

Partition a hard task into:

- a root problem statement
- explicit subgoals
- branchable uncertainty points
- verification checkpoints
- mergeable partial products

Recommended agent roles:

- `Root planner`: converts the task into subgoals, constraints, and success criteria
- `Draft branchers`: propose candidate token blocks, plans, code patches, or reasoning continuations
- `Verifier streams`: score candidate branches and return accepted prefixes or failures
- `Execution agents`: run tests, tool calls, retrieval, or lightweight symbolic checks
- `Merge agents`: reconcile accepted work into a coherent committed state

### 1.3 Meta-orchestrator

The `meta-orchestrator` is the outer control loop.

Responsibilities:

- spawn branches from unresolved uncertainty points
- assign draft, verify, and execution budget per branch
- maintain a live branch DAG
- track branch state:
  - `drafted`
  - `verifying`
  - `accepted_prefix`
  - `needs_repair`
  - `committed`
  - `pruned`
- throttle branch growth when verifier load becomes the bottleneck
- merge compatible accepted prefixes into a shared working state

### 1.4 Practical parallelism model

At this throughput regime, the useful unit is not “one giant agent per task.” It is “many short, bounded branches per task.”

For a hard coding problem, one plausible allocation is:

- `1` root planner
- `4-8` draft branchers
- `2-4` verifier streams
- `1-2` execution/test streams
- `1` merge/consensus controller

## 2. Problem Formulation

### 2.1 Speculative token decoding

Speculative token decoding predicts a token block:

- given prefix `x_<=t`
- draft proposes `x_{t+1:t+k}`
- verifier accepts the longest valid token prefix

Unit of speculation:

- token block

Acceptance means:

- the verifier accepts a prefix of drafted tokens

### 2.2 Speculative reasoning

Speculative reasoning predicts a branch of reasoning states:

- `R_i` is a reasoning state, not just a token position
- draft proposes `B_i = (R_{i+1}, ..., R_{i+m})`

Reasoning states may contain:

- natural-language rationale
- structured constraints
- code edits
- retrieved evidence
- tool outputs
- local confidence or uncertainty metadata

Unit of speculation:

- reasoning step or branch prefix

Acceptance means:

- the verifier accepts a prefix of a candidate reasoning branch as valid, useful, and consistent with current constraints

Partial acceptance is central:

- a branch may be correct through step `k`
- incorrect at `k+1`
- still useful because the accepted prefix can be committed

## 3. Draft Reasoner and Verification Oracle

### 3.1 Draft reasoner design

A pure small LM is not enough for branch-level reasoning. The strongest draft design is hybrid:

- a small LM for local continuation
- a planning layer for subgoal management
- retrieval hooks for external facts and code context
- lightweight heuristics for branch utility and repair targeting

This mirrors the logic behind DFlash-style drafting:

- keep the draft path cheap
- make it good at local futures
- rely on the large verifier for correctness

Recommended composition:

- `small LM drafter`: proposes next reasoning fragments or code deltas
- `symbolic constraint tracker`: keeps explicit goal state
- `retrieval adapter`: injects docs, APIs, code context, failing test evidence

### 3.2 Verification oracle

The large model acts as a verifier rather than as a full autoregressive re-generator.

Requirements:

- score a branch without replaying a full search from scratch
- return accepted prefix length
- identify the first failing step
- expose a branch-level consistency score

### 3.3 Verification protocol

The verifier consumes:

- current state `S`
- candidate branch steps `B = [b_1 ... b_m]`

It returns:

- per-step accept scores
- branch consistency score
- accepted prefix length
- first failing step index

```python
def verify_branch(state, branch_steps, verifier, max_prefix):
    encoded = verifier.encode(state, branch_steps[:max_prefix])

    step_scores = verifier.score_steps(encoded)
    consistency = verifier.score_consistency(encoded)
    invariants = verifier.score_invariants(encoded)

    accepted_prefix_len = 0
    for i, (s, inv) in enumerate(zip(step_scores, invariants)):
        if s < verifier.step_threshold or inv < verifier.invariant_threshold:
            return {
                "accepted_prefix_len": accepted_prefix_len,
                "step_scores": step_scores,
                "branch_score": consistency,
                "failure_index": i,
            }
        accepted_prefix_len += 1

    return {
        "accepted_prefix_len": accepted_prefix_len,
        "step_scores": step_scores,
        "branch_score": consistency,
        "failure_index": None,
    }
```

### 3.4 FP4 / MoE compatibility

This verifier path should stay compatible with an FP4 MoE serving stack by:

- verifying chunks rather than replaying long full trajectories
- routing only relevant experts per branch type
- keeping verification heads or uncertainty heads at somewhat higher precision if needed

## 4. Consensus, Forking, and Pruning

### 4.1 Consensus mechanism

Classical consensus protocols assume binary state transitions. LLM outputs are soft, structured, and partially correct.

A better fit is `Weighted Prefix Consensus`.

Each verified branch contributes:

- accepted prefix
- verifier confidence
- evidence from tests or tools
- semantic novelty
- estimated remaining completion cost

Commit decisions are made over branch fragments, not only full branches.

### 4.2 Fork vs prune rule

Use a utility score:

`Score(b) = alpha * P_mass(b) + beta * Div(b) - gamma * Cost_remain(b)`

Where:

- `P_mass(b)`: token or branch probability mass
- `Div(b)`: semantic divergence from root hypothesis
- `Cost_remain(b)`: expected compute needed to finish and verify

Interpretation:

- fork when diversity is useful and branch utility remains high
- prune when probability mass is low and remaining cost is high
- repair when the branch has a strong accepted prefix but local failure

### 4.3 Orchestration loop

```python
def orchestrate(task, budget_ms, max_active_branches):
    root = plan_root(task)
    frontier = spawn_initial_branches(root)
    committed = []

    while frontier and now_ms() < budget_ms:
        drafts = draft_more(frontier, max_active_branches)
        verified = [
            verify_branch(root.state, b.steps, verifier, b.max_prefix)
            for b in drafts
        ]

        scored = []
        for b, v in zip(drafts, verified):
            utility = branch_score(
                prob_mass=b.prob_mass,
                semantic_divergence=b.divergence,
                remaining_cost=b.remaining_cost,
                accepted_prefix=v["accepted_prefix_len"],
                branch_score=v["branch_score"],
            )
            scored.append((b, v, utility))

        committed += consensus_commit(scored)
        frontier = fork_repair_prune(scored, committed, max_active_branches)

    return merge_final(committed)
```

## 5. Training Objectives, Benchmarks, and Latency Budget

### 5.1 Training objective

Draft reasoner objective:

- maximize expected accepted prefix or accepted branch length
- penalize wasted branch compute

Verifier objective:

- learn step-level acceptance
- learn branch-level ranking
- learn process-level reward or consistency scoring

Useful signals:

- contrastive reasoning pairs
- accepted vs rejected branch prefixes
- self-play repair traces
- execution/test feedback for code
- process reward labels

### 5.2 Benchmarks

Best fits for showing reasoning-branch advantage over token-only speculation:

- `ARC`
  - branch-heavy elimination and verifier usefulness
- `MATH`
  - long reasoning chains and partial correctness
- `SWE-bench`
  - branch propose / verify / test / commit loops
- `LiveCodeBench`
  - latency-sensitive coding verification

### 5.3 Latency budget

For an interactive hard coding task, a plausible `1.5s` target:

- `100 ms` root planning
- `250 ms` first branch draft wave
- `300 ms` first verification wave
- `400 ms` tests / tool calls / repair branching
- `300 ms` second verification + consensus
- `150 ms` final merge

For harder parliament-style resolution:

- `2.5s - 5.0s`

At this point, token generation may no longer dominate. Verification, testing, merging, and branch-control overhead become the main latency source.

## 6. Failure Modes and Tradeoffs

### 6.1 Top failure modes at high speed

1. `Branch explosion`
- too many plausible branches too quickly
- mitigation:
  - active-set caps
  - utility floors
  - diversity-aware beam limits

2. `Verifier bottleneck`
- draft generation becomes cheaper than verification
- mitigation:
  - prefix verification
  - tiered verifiers
  - batch branch scoring
  - escalate only uncertain branches

3. `Fast correlated error`
- many agents make the same wrong assumption at speed
- mitigation:
  - heterogeneous drafters
  - diversity prompting
  - retrieval-conditioned branching
  - contradiction mining during consensus

### 6.2 Tradeoff summary

| Design Choice | Benefit | Cost | Best Use |
|---|---|---|---|
| Small LM draft reasoner | very fast branch generation | noisier branches | local continuation |
| Hybrid draft + planner | more structured branches | more orchestration complexity | coding and branch-heavy tasks |
| Prefix verification | avoids full replay | may miss late global failures | early pruning |
| Weighted prefix consensus | low-latency merge | softer than exact agreement | structured LLM outputs |
| High branch diversity | reduces correlated failure | more verifier load | ambiguous tasks |
| Tight pruning | lower latency | may kill useful branches | interactive mode |

### 6.3 Final thesis

DFlash-style speculative decoding shows that serial token generation can be collapsed toward near-parallel generation. The natural systems extension is to make the unit of speculation bigger: not just tokens, but reasoning branches. The long-term win is then not merely the fastest generator, but the fastest verifier-backed branch manager under tight latency and memory constraints.
