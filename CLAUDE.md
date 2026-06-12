# Project instructions

## Context management

When the conversation context is getting close to running out (approaching
auto-compaction), proactively do the following BEFORE letting compaction
happen:

1. Update `README.md` with the current state of any in-progress HPO sweeps,
   experiments, or findings (study names, job IDs, best results so far,
   open questions/bugs).
2. Update `CODE_EXPLANATION.md` if any code structure, data flow, or
   algorithm changed since it was last written.
3. Commit these doc updates (no co-author line, see below).

This keeps the docs as a durable record of progress so nothing is lost across
compaction/session boundaries, independent of the conversation summary.

## Git commits

Never add Claude/Anthropic as a co-author in git commit messages.
