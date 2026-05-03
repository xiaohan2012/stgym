#!/usr/bin/env bash
# Registers project-required plugin marketplaces and installs plugins.
# Idempotent — safe to run multiple times.
set -euo pipefail

PLUGINS_DIR="${HOME}/.claude/plugins"
MARKETPLACES_FILE="${PLUGINS_DIR}/known_marketplaces.json"

# Bootstrap known_marketplaces.json if Claude hasn't been initialised yet
if [ ! -f "$MARKETPLACES_FILE" ]; then
  mkdir -p "$PLUGINS_DIR"
  NOW="$(date -u +%Y-%m-%dT%H:%M:%S.000Z)"
  cat > "$MARKETPLACES_FILE" <<JSON
{
  "claude-plugins-official": {
    "source": {"source": "github", "repo": "anthropics/claude-plugins-official"},
    "installLocation": "${PLUGINS_DIR}/marketplaces/claude-plugins-official",
    "lastUpdated": "${NOW}"
  }
}
JSON
  echo "Initialised ${MARKETPLACES_FILE}"
fi

# Register any missing marketplaces
python3 - <<'EOF'
import json, os
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

# Install plugins (project scope, idempotent — failures are non-fatal)
install_plugin() {
  local plugin="$1"
  if claude plugins install "$plugin" --scope project 2>/dev/null; then
    echo "Installed plugin: $plugin"
  else
    echo "Plugin already installed or unavailable: $plugin"
  fi
}

install_plugin "marimo-pair@marimo-pair"
install_plugin "context7-plugin@context7-marketplace"
install_plugin "wiki-skills@wiki-skills"
