# Sandbox

Docker sandbox for running Claude Code agents (PM / Developer / Reviewer) in an isolated environment with auto-accepted permissions.

## Usage

```bash
bash sandbox/run.sh       # build + enter interactive shell in the container
bash sandbox/claude.sh    # start Claude Code inside the sandbox (interactive)
bash sandbox/claude.sh "take issue #150 and implement it"   # one-shot
```

`run.sh` extracts the Claude Code OAuth refresh token from the macOS Keychain and `GH_TOKEN` from `gh auth token`, then starts the container with the repo bind-mounted at `/repo`.

## Marimo notebooks

Port 2718 is forwarded from the container to the host. A bash wrapper automatically adds `--host 0.0.0.0` to `marimo edit` and `marimo run` so the server is reachable from outside the container. To edit a notebook from inside the sandbox:

```bash
marimo edit rct_experiment_analysis.py
```

Then open `http://localhost:2718/` in your host browser.

## Cost & Observability

Autonomous agents can burn tokens silently — loops, wasted re-reads, runaway retries. Two safeguards: **turn caps** to bound a single run, and **telemetry** to review spend after the fact.

### Turn caps (`maxTurns`)

Set in each agent's frontmatter to bound a single invocation:

- **PM** — 30 (triage is cheap; if it's hitting 30 something is stuck)
- **Developer** — 100 (implementation + tests + review + PR)
- **Reviewer** — 50 (read diff, check, comment)

If an agent consistently hits its cap, that's a signal to look at the telemetry for that session and either fix the prompt or raise the cap deliberately.

### Telemetry (OpenTelemetry)

The sandbox runs a second service alongside the agent: an **OpenTelemetry Collector** sidecar (`otel/opentelemetry-collector-contrib`). The agent emits OTel metrics + logs over OTLP/gRPC to `otel-collector:4317`; the collector batches them and writes newline-delimited JSON to `/var/log/claude-telemetry/otel.jsonl`, which is bind-mounted to `sandbox/telemetry/` on the host. Rotation (50 MB per file, 10 backups, 30 days) is configured in `sandbox/otel-collector-config.yaml`.

Architecture:

```
┌─────────┐   OTLP/gRPC    ┌────────────────┐   file        ┌──────────────────┐
│  agent  │ ─────────────> │ otel-collector │ ────────────> │ sandbox/telemetry│
│ (claude)│   :4317        │    sidecar     │  (JSONL +     │  /otel.jsonl*    │
└─────────┘                └────────────────┘   rotation)   └──────────────────┘
                                                                    ▲
                                                                    │ host bind mount
                                                         scripts/cost-report.sh
```

Why a collector instead of the console exporter: Claude Code's console exporter writes a multi-line pretty-printed JS object to stdout, which is not trivially parseable. OTLP → collector → file exporter gives us clean single-line JSON that `jq` can consume.

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

`sandbox/telemetry/` is gitignored — do not commit, paste, or attach these files anywhere public. The raw files contain:

- Bash commands the agent ran (may include env-var expansions)
- File paths and GitHub URLs touched during work
- Your prompt text

**Important:** the collector writes raw OTLP data without any scrubbing — `otel.jsonl` may contain unredacted secrets if an agent ever inlined one into a command. Scrubbing happens at **read time** inside `scripts/cost-report.sh` via a `sed` pipeline that matches common patterns (`ghp_*`, `sk-ant-*`, `sk-*`, `AKIA*`, `xox[baprs]-*`, private keys). Before sharing any excerpt from `sandbox/telemetry/` (e.g. in a bug report), run it through the report script and `grep` for `REDACTED` to confirm the scrubber fired, and still give the output a manual skim.
