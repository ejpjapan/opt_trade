#!/usr/bin/env bash

# Create or update the Conda environment declared in environment.yml.
#
# The script expects environment.yml beside this file. It uses the YAML
# `name:` value, lets Mamba choose the normal per-user environment location,
# previews updates before applying them, and prunes undeclared packages.
# If the repository contains pyproject.toml, it installs the repository in
# editable mode instead of adding the repository root to sys.path with a .pth
# file.

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
yaml_path="$project_root/environment.yml"

if ! command -v mamba >/dev/null 2>&1; then
    printf 'Error: mamba is not available on PATH.\n' >&2
    exit 1
fi

if [[ ! -f "$yaml_path" ]]; then
    printf 'Error: environment.yml not found beside %s\n' "${BASH_SOURCE[0]}" >&2
    exit 1
fi

env_name="$(awk -F: '
    /^[[:space:]]*name[[:space:]]*:/ {
        value = $2
        sub(/^[[:space:]]*/, "", value)
        sub(/[[:space:]]*$/, "", value)
        gsub(/^["\047]|["\047]$/, "", value)
        print value
        exit
    }
' "$yaml_path")"

if [[ -z "$env_name" ]]; then
    printf 'Error: no valid name: entry found in %s\n' "$yaml_path" >&2
    exit 1
fi

printf 'Environment name : %s\n' "$env_name"
printf 'YAML file        : %s\n' "$yaml_path"
printf 'Project root     : %s\n' "$project_root"

environment_exists() {
    mamba env list | awk -v target="$env_name" '$1 == target { found = 1 } END { exit !found }'
}

if ! environment_exists; then
    printf '\nEnvironment not found; creating it...\n'
    mamba env create --file "$yaml_path"
else
    printf '\nPreviewing update and prune...\n\n'
    mamba env update \
        --name "$env_name" \
        --file "$yaml_path" \
        --prune \
        --dry-run

    printf '\nProceed with update? [y/N] '
    read -r choice

    if [[ ! "$choice" =~ ^[Yy]$ ]]; then
        printf 'Aborted; no environment changes applied.\n'
        exit 0
    fi

    printf '\nApplying update...\n'
    mamba env update \
        --name "$env_name" \
        --file "$yaml_path" \
        --prune
fi

if [[ -f "$project_root/pyproject.toml" ]]; then
    printf '\nInstalling the repository in editable mode...\n'
    mamba run --name "$env_name" \
        python -m pip install --editable "$project_root"
else
    printf '\nNo pyproject.toml found; skipping editable project installation.\n'
fi

printf '\nEnvironment details:\n'
mamba run --name "$env_name" python -c \
    'import platform, sys; print(f"Python: {sys.version.split()[0]}"); print(f"Architecture: {platform.machine()}"); print(f"Executable: {sys.executable}")'

printf '\nDone. Activate with:\n  conda activate %s\n' "$env_name"
