#!/usr/bin/env bash
# Registers project-required plugin marketplaces in the local Claude Code installation.
# Run once after setting up a fresh sandbox/Docker environment.
set -euo pipefail

MARKETPLACES_FILE="${HOME}/.claude/plugins/known_marketplaces.json"

if [ ! -f "$MARKETPLACES_FILE" ]; then
  echo "known_marketplaces.json not found at $MARKETPLACES_FILE — is Claude Code installed?"
  exit 1
fi

python3 - <<'EOF'
import json, os, sys
from datetime import datetime, timezone

path = os.path.expanduser("~/.claude/plugins/known_marketplaces.json")
with open(path) as f:
    data = json.load(f)

now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
plugins_dir = os.path.expanduser("~/.claude/plugins/marketplaces")

required = {
    "marimo-pair": {
        "source": {"source": "github", "repo": "marimo-team/marimo-pair"},
        "installLocation": f"{plugins_dir}/marimo-pair",
        "lastUpdated": now,
        "autoUpdate": True,
    },
    "context7-marketplace": {
        "source": {"source": "github", "repo": "upstash/context7"},
        "installLocation": f"{plugins_dir}/context7-marketplace",
        "lastUpdated": now,
    },
    "wiki-skills": {
        "source": {"source": "github", "repo": "xiaohan2012/wiki-skills"},
        "installLocation": f"{plugins_dir}/wiki-skills",
        "lastUpdated": now,
    },
}

added = []
for name, entry in required.items():
    if name not in data:
        data[name] = entry
        added.append(name)

with open(path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

if added:
    print(f"Registered marketplaces: {', '.join(added)}")
else:
    print("All marketplaces already registered.")
EOF
