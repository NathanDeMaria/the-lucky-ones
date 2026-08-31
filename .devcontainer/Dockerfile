# 3.14.7-trixie on 2026-08-29
FROM python@sha256:48651f00145ad01e9f83d468c57cec40fac72081950f9730205b87abc6087552 AS base

# uv gets its own venv so its dependencies can't collide with the project's,
# and only the `uv` binary is linked onto PATH -- putting the whole venv's bin
# there would shadow `python` and `pip` with its copies. Root-owned
# deliberately: `dev` below only ever executes uv, it never installs into it.
#
# Pinned. Bump it on purpose: uv is what reads uv.lock, and a resolver that
# moves on its own moves the environment with it.
ENV UV_VERSION=0.8.17
RUN python3 -m venv /opt/uv \
    && /opt/uv/bin/pip install --no-cache-dir "uv==${UV_VERSION}" \
    && ln -s /opt/uv/bin/uv /usr/local/bin/uv

# The interpreter is this image's, not one uv fetches. Without this uv
# downloads its own build to satisfy `requires-python`, which on an older uv
# can mean a release candidate -- and then the container has two 3.14s, only
# one of which has the image's OpenSSL and CA bundle behind it.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3
# The workspace is a bind mount, so uv's cache (in the image, on overlayfs)
# and .venv (on the mount) are different filesystems and hardlinking between
# them fails. Copying is the fallback uv would warn its way into anyway.
ENV UV_LINK_MODE=copy

# Non-root, uid/gid 1000 to match the host user whose files get bind-mounted
# in (~/.aws, ~/.claude). Running as root here means every SSO token refresh
# clobbers the host's cache with root-owned files.
RUN apt-get update && apt-get install -y --no-install-recommends sudo \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 dev \
    && useradd --uid 1000 --gid 1000 -m -s /bin/bash dev \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev \
    && chmod 0440 /etc/sudoers.d/dev

USER dev
WORKDIR /home/dev

# Parent of the bind mounts in devcontainer.json. Docker creates a missing
# mount point as root, and chowning on the host can't fix it because the
# parent itself isn't mounted -- so `aws configure`, or anything else writing
# a new file straight into ~/.aws, would hit EPERM. Pre-creating it as dev
# means docker finds it there and leaves ownership alone.
RUN mkdir -p /home/dev/.aws

ENV PATH=/home/dev/.local/bin:${PATH}

# Deliberately no `uv sync` here. uv.lock's environment belongs at
# ${workspaceFolder}/.venv, and "the workspace" at build time is /home/dev --
# but VS Code opens the bind mount at /workspaces/the-lucky-ones, so a venv
# built here would never be the one the editor or `uv run` picks up.
# devcontainer.json's postCreateCommand does the sync in the workspace.

# The GitHub CLI, for PR and issue work from a Claude session. Debian ships no
# usable gh package and the upstream apt repo would be an unpinned external
# source, so this takes the release .deb at a pinned version instead. dpkg
# rather than apt-get because the package lists are deleted above; gh's only
# dependency is git, which the base image already has.
ARG GH_VERSION=2.97.0
RUN arch="$(dpkg --print-architecture)" \
    && curl -fsSL -o /tmp/gh.deb "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_${arch}.deb" \
    && sudo dpkg -i /tmp/gh.deb \
    && rm /tmp/gh.deb

# Only needed interactively (`aws sso login`) -- nothing in lucky_ones shells
# out to the CLI. The plays it reads live in a bucket, so a session that wants
# real data needs credentials the SSO flow puts in the mounted ~/.aws.
RUN arch="$(uname -m)" \
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${arch}.zip" -o awscliv2.zip \
    && unzip -q awscliv2.zip \
    && sudo ./aws/install \
    && rm -r awscliv2.zip ./aws

# Claude Code, pinned like uv above. Installs the native build to
# ~/.local/share/claude/versions/<v> and symlinks a launcher into
# ~/.local/bin, which the PATH above already covers. Auto-updates still run
# inside a live container; this only pins what a fresh build starts from.
RUN curl -fsSL https://claude.ai/install.sh | bash -s 2.1.235

# Claude's top-level config lives at $HOME/.claude.json, which is *outside*
# the ~/.claude bind mount -- so a rebuild loses it and the first `claude` run
# replays the onboarding prompts. Seeding these two keys skips that; Claude
# fills in the rest of the file on first run.
RUN printf '%s' '{"hasCompletedOnboarding":true,"theme":"dark"}' \
    > /home/dev/.claude.json

# For persisting bash history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" && echo $SNIPPET >> "/home/dev/.bashrc"

# Machine-local environment -- the AWS_PROFILE to use, mainly. It has to be
# settable without committing it, so it lives in .devcontainer/local.env,
# which .gitignore and .dockerignore both drop; devcontainer.json passes its
# path down as DEVCONTAINER_ENV_FILE (from ${containerWorkspaceFolder}, since
# where the workspace lands depends on which folder you opened).
#
# Sourced from two places on purpose, because bash reads different files
# depending on how it was started and both cases matter here:
#   .bashrc    interactive shells -- a VS Code terminal
#   BASH_ENV   non-interactive ones -- `bash -c`, which is how Claude Code and
#              anything scripted runs commands
# Keep this script silent: BASH_ENV is read before *every* non-interactive
# bash, so anything it prints lands in the output of every `$(...)` capture.
COPY --chown=dev:dev .devcontainer/local-env.sh /home/dev/.local-env.sh
ENV BASH_ENV=/home/dev/.local-env.sh
RUN echo '. /home/dev/.local-env.sh' >> "/home/dev/.bashrc"

COPY --chown=dev:dev profile.sh /home/dev/profile.sh
RUN echo ". profile.sh" >> "/home/dev/.bashrc"
