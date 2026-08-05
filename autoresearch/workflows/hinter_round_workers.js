export const meta = {
  name: 'rlad-hinter-round-workers',
  description: 'Run the ten RLAD hint-worker subagents concurrently for one research round',
  phases: [
    { title: 'Workers', detail: 'ten independent per-hint train+propose workers' },
  ],
}

const ROUND = args.round
const RRR = String(ROUND).padStart(3, '0')
// Round DIRECTORIES are 3-digit (research/rounds/001/) but pool TASK IDs are
// 2-digit: core.ROUND_ID_PATTERN = (?:0[1-9]|1[0-9]|20), so `r01-train-h01` is
// valid and `r001-train-h01` is rejected by validate_task_identity. Round 1's
// ten workers each hit this and self-corrected; keep both forms distinct.
const RR = String(ROUND).padStart(2, '0')
const WORK = 'work_zsw_lambda10'
const ENVPREFIX = 'RLAD_REPO_ROOT="$PWD" RLAD_AUTORESEARCH_WORK="$PWD/work_zsw_lambda10" RLAD_AUTORESEARCH_LAMBDA=10 RLAD_AUTORESEARCH_PARTITION="ml.p5.48xlarge" RLAD_AUTORESEARCH_NODES="ip-10-1-38-11,ip-10-1-81-8"'

const HINTS = ['01','02','03','04','05','06','07','08','09','10']

const RESULT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['hint_id', 'wrote_proposal', 'execution_id', 'gpu_count', 'train_correct', 'mutation', 'note'],
  properties: {
    hint_id: { type: 'integer' },
    wrote_proposal: { type: 'boolean' },
    execution_id: { type: 'string' },
    gpu_count: { type: 'integer' },
    train_correct: { type: 'integer' },
    mutation: { type: 'string' },
    note: { type: 'string' },
  },
}

function workerPrompt(ii) {
  const hintId = parseInt(ii, 10)
  return `You are worker ${hintId} of 10 in round ${ROUND} of the RLAD 10-hint autoresearch
experiment. You own EXACTLY ONE hint: hint_${ii} (hint_id=${hintId}). Your job is to
produce one improved replacement for that hint, justified by fresh GPU training
evidence that you must obtain yourself.

Working directory: /fsx/gstevenw/testing_alignment_algos/RLAD (the repo root). Use absolute
paths or cd there first. Shell state does NOT persist between your Bash calls,
so prefix EVERY autoresearch command with the env vars shown below.

## Objective you are optimizing

This run uses lambda = 10:

    J_i = train_i + 10 * heldout_i

train_i   = fraction of your 8 training rollouts the grader marks correct
heldout_i = your hint's AGGREGATE accuracy over 10 hidden held-out questions

Because lambda is 10, held-out transfer dominates: a hint that memorizes your
one training problem but does not generalize will LOSE. Your incumbent will be
replaced only if the proposal scores higher J (tie-break: higher held-out, then
shorter). So write a REUSABLE STRATEGY, not a solution.

## What eighteen rounds of evidence say (read this first)

Keep counts: r1 5, r2 0, r3 5, r4 2, r5 3, r6 2, r7 0, r8 1, r9 0, r10 1, r11 1,
r12 1, r13 0, r14 1, r15 1, r16 1, r17 2 (out of 10 each). Book mean J:
2.0125 -> 2.7875; book mean held-out 0.165 -> 0.22875.

THE SEARCH HAS SATURATED UNDER SMALL EDITS. Roughly 7 of the last 80 proposals
survived. Your incumbent has repelled many consecutive challengers. But note the
recent trend is encouraging: rounds 14-17 each kept something, and rounds 16 and
17 produced genuine held-out GAINS (hint 10: 0.1625 -> 0.2250; hint 9: 0.2625 ->
0.2875), not just tie-break wins. Real improvements are still available.

SAMPLING IS NOT DETERMINISTIC - do not assume otherwise. A round-18 worker argued
that because config.json pins seed=1234, an unchanged hint must re-score
identically, and concluded the "training is noisy" advice was wrong. I checked it
directly against every repeated measurement in this run: of the 25 hint texts
trained more than once, 21 scored DIFFERENT train_correct on re-measurement with
byte-identical text. Hint 1's incumbent was measured 17 times and scored 0, 1, 2
and 3 out of 8; hint 9's was measured 10 times and scored 0, 1, 3 and 4. A pinned
seed does not make batched vLLM generation reproducible. So treat your manifest's
train numbers as noisy samples, never as exact values, and never conclude a text
would re-score the same. (The private/held-out side never re-measures the same
text at all, which is why it can look deterministic - it is simply unrepeated.)

Two practical consequences:

FIRST, a STRICTLY LOSSLESS COMPRESSION is the highest-expected-value move for
most hints. Same steps, same meaning, same order, fewer tokens. It cannot raise
transfer, but it converts a held-out tie into a win, and that is how every
recent keep happened. Do it carefully: the compressions that LOST all altered a
content-bearing phrase while believing they were lossless. Change function words
and redundant syntax only; keep every content phrase byte-identical if you can.

SECOND, semantic edits need a genuinely new idea, not a new phrasing of an old
one. Before proposing one, check your manifest: if a previous round already
tested the same underlying idea and lost, do not re-test it with different
wording - that has happened repeatedly and lost every time.

WHERE THE REMAINING HEADROOM IS. Held-out per hint is now: h1 0.250, h2 0.225,
h3 0.225, h4 0.225, h5 0.250, h6 0.250, h7 0.175, h8 0.163, h9 0.263, h10 0.150.
If you are hint 7, 8 or 10 your transfer is well below the rest of the book, so a
genuine strategy improvement is both more plausible and more valuable for you -
prefer Option A and be ambitious. If you are hint 1, 2, 5, 6 or 9 you are at or
near the book ceiling; prefer Option B and protect what works.

The important consequence: SMALL SAFE-LOOKING EDITS ARE NOT SAFE. They are the
single most-tested and most-failed category in this book. Specifically these have
all been tried repeatedly and lost:
  - swapping one vague clause for a sharper-sounding one
  - deleting a clause your rollouts never visibly execute (round 8: lost 0.2250
    -> 0.1375 doing exactly this)
  - trimming a token or two with no change of meaning
  - changing a closing verb, or re-ordering steps
  - adding a caution, prohibition, slogan, or problem-specific trap rule

So do not spend this round on a cosmetic tweak. You have two genuinely useful
options, and picking either honestly is fine:

  OPTION A - PROPOSE A DIFFERENT STRATEGY. Ask what a strong solver would
  actually DO on this class of problem, and whether your hint names that. The
  book's real wins all came from naming a concrete executable move that replaced
  a vague gesture (hint 9 round 1; hint 5 round 6, J 2.50 -> 3.25) or from
  replacing a directive the failures visibly follow into a wall (hint 6 round 8,
  J 2.125 -> 2.750 after seven straight losses). If your rollouts keep dying the
  same way, ask whether your hint's core approach is itself the problem - not
  whether its wording can be polished.

  OPTION B - PROPOSE A CAREFUL LOSSLESS COMPRESSION. Same strategy, same steps,
  provably nothing dropped, fewer tokens. This wins ties and has won outright
  (hint 2 round 3: 81 -> 70 tokens, J 3.00 -> 3.25).

Whichever you choose, state honestly in your summary how likely you think it is
to beat the incumbent and why. A well-reasoned near-miss with an accurate
self-assessment is a good outcome; the incumbent surviving is not a failure.

Findings that hold across all ten hints:

1. HELD-OUT DECIDES; THE 8-ROLLOUT TRAINING SIGNAL IS NOISE. Unchanged incumbents
   have scored wildly different fresh training values between rounds (one hint
   went 0/8, 4/8, 3/8, 4/8, 0/8 with no edit at all; another went 5/8 then 8/8).
   Held-out is 80 samples and is weighted 10x.
   => Never redesign your hint around one round's training miss. Target the
   SYSTEMATIC weakness visible repeatedly across your own history.

2. LENGTH IS A REAL COST, BUT SHORTENING IS NOT A FREE WIN. Rounds 2 and 4 proved
   the cost: every proposal longer than its incumbent lost (all ten in round 2).
   But round 6 tested the converse - nine of ten proposals were SHORTER and only
   two survived, several losing held-out badly (one fell 0.2500 -> 0.0875).
   => Length is a tiebreaker and a tax, not the objective. Do not cut text merely
   to be short. Cut only text you have EVIDENCE is inert (see 3), and never cut a
   step the successful rollouts actually rely on - that is what caused round 6's
   biggest losses.

3. THE HIGHEST-VALUE EDIT IS REMOVING A CLAUSE THE ROLLOUTS DEMONSTRABLY IGNORE,
   OR NAMING A CONCRETE FIRST MOVE THEY LACK. Read your eight rollouts and check
   which of your hint's clauses actually appear in their reasoning. A clause no
   rollout ever executes is pure transfer tax. A vague gesture ("look for a
   symmetry") that you can replace with one concrete, executable instruction has
   produced the book's biggest gains. Wording a 1.7B student will act on beats
   wording that is merely correct.

4. WHAT HAS RELIABLY FAILED: appending caution lists, prohibitions, slogans,
   problem-specific "trap rules", and generic multi-step procedure checklists.
   These buy training accuracy and destroy transfer. If your only idea is to warn
   the student about the mistake you just watched it make, that has lost across
   the book many times - prefer a positive, constructive instruction instead.

Your incumbent has already survived several challenges, so the bar is high. An
honest small improvement beats an ambitious rewrite.

## Absolute rules (violating any of these invalidates the round)

- You may read ONLY these files:
  * ${WORK}/research/rounds/${RRR}/training_inputs/hint_${ii}.json   (your public packet)
  * ${WORK}/research/rounds/${RRR}/worker_history/hint_${ii}.json    (your manifest)
  * every same-question artifact EXPLICITLY LISTED inside that manifest
  * your own training output/receipt/log for the task you launch below
- NEVER read another worker's files (any hint_XX where XX != ${ii}), any private
  evaluator log, the source dataset, or the HF dataset cache.
- NEVER try to see held-out problems, answers, rollouts, or per-question
  rewards. Only AGGREGATE heldout_i / J_i numbers are legitimate for you.
- NEVER run Qwen/vLLM inference yourself or on the login node. All inference
  goes through the Slurm pool queue described below.
- Your hint must be <= 200 Qwen tokens and MUST differ from the incumbent.
- Hints must be reusable strategy. No final numeric answers, no full worked
  solution to your specific problem.

## Step 1 - read your inputs

Read your packet and your manifest:

    ${WORK}/research/rounds/${RRR}/training_inputs/hint_${ii}.json
    ${WORK}/research/rounds/${RRR}/worker_history/hint_${ii}.json

The packet gives you: problem, answer, your current hint, and your previous
train_i / heldout_i / J_i. The manifest lists your held-out-safe history for
this same question (prior rounds' rollouts, proposals, decisions, and
aggregate-only private scores). Read every artifact the manifest lists - that
history tells you which mutation styles already helped or failed for THIS
question, so you do not repeat a losing edit.

## Step 2 - launch your own one-GPU training task

Enqueue your training task (this is the ONLY sanctioned way to get rollouts):

    cd /fsx/gstevenw/testing_alignment_algos/RLAD && ${ENVPREFIX} uv run --project autoresearch --frozen python -m autoresearch.hinter.pool enqueue --task-id r${RR}-train-h${ii} --mode train --input ${WORK}/research/rounds/${RRR}/training_inputs/hint_${ii}.json --output ${WORK}/research/rounds/${RRR}/training_outputs/hint_${ii}.json --receipt ${WORK}/research/rounds/${RRR}/training_receipts/hint_${ii}.json

If it reports a task already exists / is reused, that is FINE - do not
re-enqueue, just proceed to wait.

Then block until it finishes (it takes several minutes; use a long timeout of
at least 3600000 ms and do NOT poll in a sleep loop):

    cd /fsx/gstevenw/testing_alignment_algos/RLAD && ${ENVPREFIX} uv run --project autoresearch --frozen python -m autoresearch.hinter.pool wait --task-id r${RR}-train-h${ii}

Never infer results locally - you must let the GPU task produce them. If the
wait returns state=failed, read ONLY your own task log
(${WORK}/logs/tasks/r${RR}-train-h${ii}.err) and report the failure in your
result; do not fabricate a proposal.

## Step 3 - verify one GPU and study YOUR rollouts

Read your receipt ${WORK}/research/rounds/${RRR}/training_receipts/hint_${ii}.json
and CONFIRM it says gpu_count == 1. Record its execution_id.

Read your output ${WORK}/research/rounds/${RRR}/training_outputs/hint_${ii}.json
and inspect ALL EIGHT rollouts and their rewards. This is your evidence. Look
for the actual failure mode, e.g.:
  - does the student misread the setup or drop a constraint?
  - does it pick a doomed representation, or thrash between approaches?
  - does it get the right method but botch arithmetic/algebra?
  - does it run out of room, never committing to a final boxed answer?
  - does the current hint push it toward something unhelpful or too vague?

## Step 4 - write exactly one proposal

Write EXACTLY ONE JSON object to:

    ${WORK}/research/rounds/${RRR}/worker_proposals/hint_${ii}.json

with EXACTLY these five keys and nothing else:

    hint_id                 integer ${hintId}
    hint                    your revised hint text (<=200 Qwen tokens, differs from incumbent)
    mutation                short label for what you changed, e.g. "sharpen-representation-choice"
    subagent_summary        1-3 sentences: the failure mode you saw in your 8 rollouts and why this edit should transfer. Do NOT quote full rollouts.
    sampling_slurm_job_id   the receipt's execution_id, as a string

HARD LENGTH LIMITS enforced by collect-proposals. Exceed either and the ENTIRE
round is rejected, so count characters (not words) before writing:

    subagent_summary   <= 1000 characters  (aim for <= 800)
    mutation           <=  500 characters  (a short hyphenated label)

Write it with a small Python snippet using json.dump so the file is valid JSON
(a heredoc is fine, but avoid shell quoting pitfalls - verify by reading it
back and json.load-ing it).

WRITE-ONCE RULE (important). Write this file exactly ONCE, at the very end, and
never rewrite it afterwards. In rounds 4 and 5 more than one agent instance ran
per hint slot and they overwrote each other, so the version the gate validated
was sometimes a LONGER hint than the one the worker had carefully compressed -
and those longer versions lost. Therefore:
  - If no file exists at your path, write yours once and stop.
  - If a file ALREADY exists, read it first. If it is already valid (exactly the
    five keys, <=200 tokens, no answer language) AND no longer than your
    incumbent AND it cites your execution_id, LEAVE IT ALONE and say so in your
    note. Only overwrite when it is invalid or longer than the incumbent.
This makes the outcome of a race the shorter valid hint rather than the last one
written.

Guidance for a hint that wins at lambda=10: make it a transferable procedure
for this CLASS of problem - how to set up, which representation/invariant to
reach for, which check to run before committing, and a reminder to produce a
final boxed answer. Keep it concrete enough to change behavior but general
enough to help ten unseen problems. Avoid naming your problem's specific
numbers or its answer.

Token budget note: 200 Qwen tokens is roughly 130-150 English words. Stay
comfortably under it.

## Step 5 - report

Return the structured result: hint_id, wrote_proposal, execution_id, gpu_count,
train_correct (how many of your 8 rollouts were correct), mutation, and a short
note. If anything blocked you, say so honestly in note rather than inventing
evidence.`
}

phase('Workers')

const results = await parallel(
  HINTS.map((ii) => () =>
    agent(workerPrompt(ii), {
      label: `r${RR}-worker-h${ii}`,
      phase: 'Workers',
      schema: RESULT_SCHEMA,
    })
  )
)

const ok = results.filter(Boolean)
log(`round ${ROUND}: ${ok.length}/10 workers returned`)
return {
  round: ROUND,
  returned: ok.length,
  results: ok,
}
