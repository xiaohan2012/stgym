FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    curl \
    ripgrep \
    procps \
    jq \
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

# Claude Code CLI (standalone installer, same PATH as uv)
RUN curl -fsSL https://claude.ai/install.sh | bash

# Worktree directory
RUN mkdir -p /repo-worktrees

# Non-root user (required by claude --dangerously-skip-permissions)
RUN useradd -m -s /bin/bash agent \
    && chown -R agent:agent /repo-worktrees
ENV HOME=/home/agent

# Make root-installed tools accessible to agent user
RUN chmod o+x /root && chmod -R o+rx /root/.local

# Dedicated virtualenv so we don't overwrite host's .venv
ENV UV_PROJECT_ENVIRONMENT=/opt/stgym-venv
ENV UV_LINK_MODE=copy
RUN mkdir -p "$UV_PROJECT_ENVIRONMENT" \
    && chown -R agent:agent "$UV_PROJECT_ENVIRONMENT" /home/agent
USER agent
RUN uv venv "$UV_PROJECT_ENVIRONMENT"
USER root
ENV VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

# Activate venv in every bash session (for docker exec)
RUN echo "source $UV_PROJECT_ENVIRONMENT/bin/activate" >> /home/agent/.bashrc

# Git identity for agent user
RUN su agent -c 'git config --global user.name "stgym-bot" \
    && git config --global user.email "stgym-bot@users.noreply.github.com"'

# Marimo config: bind to all interfaces so the server is reachable from the host
RUN mkdir -p /home/agent/.config/marimo \
    && printf '[server]\nhost = "0.0.0.0"\n' > /home/agent/.config/marimo/marimo.toml \
    && chown -R agent:agent /home/agent/.config

# SSH config for agent user
RUN mkdir -p /home/agent/.ssh && chmod 700 /home/agent/.ssh \
    && echo "Host cyy2\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null" > /home/agent/.ssh/config \
    && chmod 600 /home/agent/.ssh/config \
    && chown -R agent:agent /home/agent/.ssh

WORKDIR /repo

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER agent
ENTRYPOINT ["/entrypoint.sh"]
