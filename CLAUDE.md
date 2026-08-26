# Working agreement

## Decisions belong to the user

Propose, then wait.
Before writing or running anything substantial, describe the plan and why, then stop and wait for approval.
Do not chain several steps together because the next one seems obvious.

The user owns these and decides them, not Claude:

- The research question and any change to its scope.
- All modelling decisions: architecture, hyperparameters, what enters the model, what gets ablated, what gets dropped.
- The blog prose.

Claude's job is to lay out options with tradeoffs, implement the choice the user makes, supply numbers, and check facts.

When a decision has already been made unilaterally, surface it explicitly so the user can review or reverse it.
Do not let an unreviewed default harden into a project assumption.

Do not run cheap previews or exploratory side tests as a substitute for doing the work properly.
The user has asked for the full version, not a fast approximation.

## Style

Never use an em dash or en dash.
Use a plain hyphen.

Never add a `Co-Authored-By` trailer to a commit.
No agent attribution in git history.

In Markdown files, put each full sentence on its own physical line.
Preserve headings, lists, and code fences as normal.

Write documentation so each section stands alone.
Name the exact file, function, or assumption rather than writing "it" or "this".
Keep documentation short, punchy, active, and free of redundancy.

## Engineering

Give little weight to development cost.
Prefer quality, simplicity, robustness, scalability, and long term maintainability.

Follow Clean Code.
Small functions doing one thing, intention-revealing names, no dead code, no comment where a rename would do.

Fix lint errors, test failures, and flaky tests on sight, even when unrelated to the current task.

## Read these before working

`CONVENTIONS.md` carries terminology, architecture constraints, and how changes are explained.
`DECISIONS.md` is an append-only log of what has been settled, with dates.
A decision there is current only if no later entry supersedes it.
`REJECTED.md` lists approaches already ruled out and why.
Read `REJECTED.md` before proposing an approach, so a ruled-out idea is not raised again.

Every settled decision gets an entry in `DECISIONS.md`.
Every ruled-out approach gets an entry in `REJECTED.md`.

## Project state

`README.md` covers the collector and how to run it.
`API_FINDINGS.md` records measured Clash Royale API behaviour.
`ASSUMPTIONS.md` records ten identifying assumptions and which are testable.
`RESULTS_BASELINE_LADDER.md` and `RESULTS_NEURAL.md` record findings.

Read `ASSUMPTIONS.md` before drawing any conclusion from the data.
