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

## What five rounds of evidence say actually wins (read this first)

Book-wide keep counts: round 1 = 5/10, round 2 = 0/10, round 3 = 5/10,
round 4 = 2/10, round 5 = 3/10. Book mean J: 2.0125 -> 2.2625 -> 2.2625 ->
2.3750 -> 2.4000 -> 2.5375.

Two findings are robust across all ten hints:

1. LENGTH IS THE DOMINANT PREDICTOR OF LOSING. In round 2 every one of the ten
   proposals was longer than its incumbent and ALL TEN were discarded. In round 4
   every proposal longer than its incumbent lost, including two of the book's
   three best hints. The clearest single win in the book was a LOSSLESS
   COMPRESSION: hint 2 went 81 -> 70 tokens with every step preserved and J rose
   3.00 -> 3.25. Because lambda multiplies held-out by 10, appended caution
   lists, prohibitions, slogans and problem-specific "trap rules" reliably
   destroy transfer even when each is individually true.
   => Treat every added token as a cost you must justify. Prefer a proposal at or
   below your incumbent's length. Deleting inert text is a legitimate and often
   winning mutation. Do not pad.

2. THE 8-ROLLOUT TRAINING SIGNAL IS NOISE-DOMINATED; HELD-OUT DECIDES. Unchanged
   incumbents have scored wildly different fresh training values between rounds
   (one hint went 0/8, 4/8, 3/8, 4/8, 0/8 with no edit at all; others went 7/8
   then 2/8, or 8/8 then 5/8). Held-out is 80 samples and is weighted 10x.
   => Do NOT redesign your hint around a single round's training miss. Target the
   SYSTEMATIC, transferable weakness visible repeatedly across your history.

Useful diagnostic: check whether the rollouts actually DO what your hint says.
Several hints contained clauses no rollout ever executed - that text is pure
transfer tax and should be cut. Wording a 1.7B student will act on beats wording
that is merely correct.

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
