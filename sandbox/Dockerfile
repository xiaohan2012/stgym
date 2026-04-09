FROM --platform=linux/arm64 python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    curl \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

# gh CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Claude Code CLI (standalone installer)
RUN curl -fsSL https://claude.ai/install.sh | sh
ENV PATH="/root/.claude/bin:$PATH"

# Git identity for agents
RUN git config --global user.name "stgym-bot" \
    && git config --global user.email "stgym-bot@users.noreply.github.com"

# SSH config: skip host key verification for cyy2
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh \
    && echo "Host cyy2\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null" > /root/.ssh/config \
    && chmod 600 /root/.ssh/config

# Worktree directory
RUN mkdir -p /repo-worktrees

# Dedicated virtualenv (not .venv/ in repo)
ENV VIRTUAL_ENV=/opt/stgym-venv
RUN uv venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /repo

# Install dependencies at build time (requires repo to be mounted or copied)
# At runtime, run: uv sync --group dev
