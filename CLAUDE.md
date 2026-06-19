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
