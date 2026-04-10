# Sandbox

Docker sandbox for running Claude Code agents (PM / Developer / Reviewer) in an isolated environment with auto-accepted permissions.

## Usage

```bash
bash sandbox/run.sh       # build + enter interactive shell in the container
bash sandbox/claude.sh    # start Claude Code inside the sandbox (interactive)
bash sandbox/claude.sh "take issue #150 and implement it"   # one-shot
```

`run.sh` extracts the Claude Code OAuth refresh token from the macOS Keychain and `GH_TOKEN` from `gh auth token`, then starts the container with the repo bind-mounted at `/repo`.

## Cost & Observability

Autonomous agents can burn tokens silently — loops, wasted re-reads, runaway retries. Two safeguards: **turn caps** to bound a single run, and **telemetry** to review spend after the fact.

### Turn caps (`maxTurns`)

Set in each agent's frontmatter to bound a single invocation:

- **PM** — 30 (triage is cheap; if it's hitting 30 something is stuck)
- **Developer** — 100 (implementation + tests + review + PR)
- **Reviewer** — 50 (read diff, check, comment)

If an agent consistently hits its cap, that's a signal to look at the telemetry for that session and either fix the prompt or raise the cap deliberately.

### Telemetry (OpenTelemetry)

The sandbox enables Claude Code's OTel output via env vars in `sandbox/docker-compose.yml`. Events (metrics + per-turn records) are written via the console exporter to `/var/log/claude-telemetry/otel-<timestamp>.jsonl` inside the container, which is bind-mounted to `sandbox/telemetry/` on the host. `sandbox/claude.sh` generates a fresh log file per invocation and runs output through a secret-scrubbing `sed` filter on the way in.

**What's logged (`OTEL_LOG_TOOL_DETAILS=1`, `OTEL_LOG_USER_PROMPTS=1`):**
- Per-turn cost, tokens (input/output/cache), duration, model
- Tool calls: tool name, inputs (bash command, file path, URL), duration, success/failure
- User prompts (text)
- API errors, retries, rate limits

**What's NOT logged:**
- Tool outputs (file contents, command stdout/stderr, response bodies) — `OTEL_LOG_TOOL_CONTENT` is intentionally off
- Model responses

### Weekly review

```bash
bash scripts/cost-report.sh          # last 7 days
bash scripts/cost-report.sh 30       # last 30 days
```

Prints per-session totals (turns, cost, tokens, error count, first prompt) sorted by cost, plus grand totals. Look for:

- Sessions near the `maxTurns` ceiling → agent is stuck or prompt is vague
- Input tokens trending up session-over-session → context bloat
- `api_error` count > 0 → rate limits or model errors worth checking

For ad-hoc spot-checks inside an interactive session, use the `/cost` slash command.

### Security: never share telemetry files

`sandbox/telemetry/` is gitignored — do not commit, paste, or attach these files anywhere public. Even with the scrub filter, the logs contain:

- Bash commands the agent ran (may include env-var expansions)
- File paths and GitHub URLs touched during work
- Your prompt text

The scrub filter redacts common secret patterns (`ghp_*`, `sk-ant-*`, `AKIA*`, private keys, etc.) but it's not exhaustive. Before sharing any output from `sandbox/telemetry/` (e.g. in a bug report), `grep` for `REDACTED` to confirm the scrubber ran, and give the file a manual skim.
