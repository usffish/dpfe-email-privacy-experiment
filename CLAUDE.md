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

## Model and effort level guide

| Task type | Model | Effort | Approach |
|---|---|---|---|
| Read a file, quick grep, git status | Sonnet | Low | Direct tool call, no subagent |
| Edit a file, update docs, small code fix | Sonnet | Low | Direct, inline |
| Understand a result / answer a question | Sonnet | Low | Inline, no subagent |
| Broad codebase exploration (>3 searches) | Sonnet | Low | Spawn Explore subagent |
| Loop / auto-drive monitoring | Sonnet | Low | ScheduleWakeup + SSH via socket |
| Debugging unexpected behavior / tracing a bug | Sonnet | Medium | Read all relevant files first |
| Multi-file refactor or complex new feature | Sonnet | Medium | Inline, read all affected files first |
| Architecture or experiment design decisions | Sonnet | High | Think through tradeoffs explicitly |
| Independent code review / second opinion | Sonnet | Medium | Spawn code-reviewer subagent |

**Effort level** controls how much reasoning Claude does before acting:
- **Low** — act immediately; no extended thinking needed for mechanical tasks
- **Medium** — think before acting; good for bugs or multi-step changes where a wrong first move wastes time
- **High** — extended reasoning; only for open-ended design questions with real tradeoffs

**Never need Opus for this project** — all tasks are code edits, SSH commands, JSON parsing,
and doc updates. Opus adds latency and cost with no quality benefit here.

**Subagents**: only spawn for open-ended codebase searches or genuinely parallel independent
work. Don't spawn to do something you can do in 1-2 tool calls inline.

## CIRCE SSH access

CIRCE only accepts Kerberos/password auth — SSH key login is disabled. However, the local
`~/.ssh/config` uses **ControlMaster multiplexing** with `ControlPersist 8h`:

```
Host circe
    HostName circe.rc.usf.edu
    User ismailj
    PreferredAuthentications password
    PubkeyAuthentication no
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h:%p
    ControlPersist 8h
```

When the user has CIRCE open in a terminal, a shared socket exists at
`~/.ssh/sockets/ismailj@circe.rc.usf.edu:22`. All automated SSH commands MUST reuse this
socket with:

```bash
ssh -o ControlMaster=no -o ControlPath=~/.ssh/sockets/%r@%h:%p circe "<command>"
```

`-o ControlMaster=no` tells SSH to be a client (not a new master), while
`ControlPath` points to the user's existing authenticated socket.
**Do NOT use `-o BatchMode=yes`** — it bypasses the socket and falls back to direct auth
(which fails). Do NOT use `-i ~/.ssh/circe_key` for login (that key is for GitHub, not CIRCE).

If the socket is missing (user closed their terminal or VPN dropped): stop and ask the user
to reconnect CIRCE in their terminal. The socket appears as soon as they SSH in.

**Always run CIRCE commands yourself** via the socket above. Never ask the user to paste
commands or output — only fall back to asking if the socket is confirmed unavailable.
