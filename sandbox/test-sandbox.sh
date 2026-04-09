#!/bin/bash
# Test plan for the agent sandbox.
# Run this inside the container: bash sandbox/test-sandbox.sh

set -euo pipefail

PASS=0
FAIL=0

check() {
    local name="$1"
    shift
    if "$@" > /dev/null 2>&1; then
        echo "  PASS: $name"
        ((PASS++))
    else
        echo "  FAIL: $name"
        ((FAIL++))
    fi
}

echo "=== Agent Sandbox Test Plan ==="
echo

# 1. Git identity
echo "[1/8] Git identity"
check "user.name is stgym-bot" test "$(git config user.name)" = "stgym-bot"
check "user.email is set" test -n "$(git config user.email)"

# 2. Claude Code
echo "[2/8] Claude Code"
check "claude CLI available" claude --version

# 3. gh CLI
echo "[3/8] GitHub CLI"
check "gh CLI available" gh --version
check "gh authenticated" gh auth status

# 4. SSH to cyy2
echo "[4/8] SSH to cyy2"
check "ssh to cyy2" ssh -o BatchMode=yes -o ConnectTimeout=5 cyy2 echo ok

# 5. Python / virtualenv
echo "[5/8] Python environment"
check "python available" python --version
check "virtualenv is /opt/stgym-venv" test "$VIRTUAL_ENV" = "/opt/stgym-venv"
check "uv available" uv --version

# 6. Unit tests
echo "[6/8] Unit tests"
check "pytest passes" pytest tests/ -m 'not slow' --tb=short -q

# 7. Pre-commit
echo "[7/8] Pre-commit / linting"
check "pre-commit passes" pre-commit run --all-files

# 8. Git worktree
echo "[8/8] Git worktree"
check "create worktree" git worktree add /repo-worktrees/test-sandbox -b test-sandbox-branch origin/main
check "remove worktree" git worktree remove /repo-worktrees/test-sandbox
git branch -D test-sandbox-branch 2>/dev/null || true

echo
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
