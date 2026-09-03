# Autonomous Prompt Files

`autonomous-action.sample.sh` (installed as `autonomous-action.sh`) builds the
heartbeat prompt with three `@FILE` mentions:

```text
@SOUL.md
@TODO.md
@ROUTINES.md
```

`@FILE` is Claude Code's context mention: the file's contents are expanded into
the prompt. The files live in the project directory, next to the script
(the script `cd`s there before running `claude -p`). None of them ship with the
repository, because their contents are yours.

## What each file is for

| File | Purpose | Who writes it |
|---|---|---|
| `SOUL.md` | The agent's personality and values: name, voice, what it cares about, what it refuses. The post-compaction recovery hook re-reads it first ("remember who you are"). | You, once; the agent may revise with you |
| `TODO.md` | What the agent wants to do. Read to pick an action, written back with what was finished. When a heartbeat runs out of `[CONTINUE]` chains it parks the rest under a "前回の続き" (carried over) section here. | You and the agent |
| `ROUTINES.md` | Recurring things with an interval and a last-run date. On a routine heartbeat the agent picks the one furthest behind schedule, runs it, and updates the date. Optional. | You and the agent |

Without them the heartbeat still runs on `CLAUDE.md` alone.

## What happens when one is missing

An `@` mention of a file that does not exist resolves to nothing, and Claude
Code does not say so. The heartbeat completes, the log looks normal, and the
prompt has been running with a dead reference the whole time. Three things now
point at it:

- `autonomous-action.sh` checks for each of the three files before building
  the prompt and writes `WARN: SOUL.md not found in <dir>; the @SOUL.md
  reference in the prompt will not resolve` to its log and to stderr. It does
  not abort.
- `scripts/doctor.sh` warns per missing file when `autonomous-action.sh` is
  installed, and merely notes their absence before that.
- This page exists.

## What the script needs on PATH

`autonomous-action.sh` shells out to two commands that nothing else in the
repository requires. Neither absence stops the heartbeat, and only one of the
four call sites says anything:

| Command | Used for | If it is missing |
|---|---|---|
| `jq` | Collecting `permissions.allow` out of `.claude/settings*.json` | `ALLOWED_TOOLS` comes back empty, `--allowedTools` is not passed to `claude -p` at all, and the `--- ALLOWED_TOOLS ---` block in the log is empty |
| `jq` | Reading `.session_id` out of the result JSON | The session file is not updated, so the next heartbeat starts a new session instead of resuming |
| `bun` | `scripts/desire-tick.ts` | stderr is captured and logged as `[欲望エラー]` -- **the one that says so** |
| `bun` | `scripts/interoception.ts` | stderr goes to `/dev/null`; the interoception line is simply absent |

The session one is the quiet one. Nothing in the log distinguishes "resumed the
previous session" from "started a fresh one because `jq` was not there", and a
heartbeat that never resumes looks like a heartbeat with a short memory.

## Templates

Minimal, neutral starting points are in `examples/`:

```bash
cp examples/SOUL.sample.md SOUL.md
cp examples/TODO.sample.md TODO.md
cp examples/ROUTINES.sample.md ROUTINES.md   # optional
```

Edit them before the first heartbeat. `presets/` is something else: those are
`~/.claude/CLAUDE.md` personality templates, not these files.
