#!/bin/zsh
set -euo pipefail

project_dir="${0:A:h}"
tws_app="${OPT_TRADE_TWS_APP:-$HOME/Applications/Trader Workstation/Trader Workstation.app}"
export PATH="$HOME/.docker/bin:/Applications/Docker.app/Contents/Resources/bin:$PATH"

fail() {
    print -u2 -- "Option Stream: $1"
    exit 1
}

[[ -d "$tws_app" ]] || fail "Trader Workstation was not found at $tws_app"
docker_bin="$(command -v docker || true)"
[[ -n "$docker_bin" ]] || fail "Docker Desktop is not installed"

print -- "Opening Trader Workstation. Log in and leave its API enabled."
/usr/bin/open "$tws_app" || fail "Could not open Trader Workstation"

if ! "$docker_bin" info --format '{{.ServerVersion}}' >/dev/null 2>&1; then
    print -- "Starting Docker Desktop..."
    "$docker_bin" desktop start --timeout 120 || fail "Docker Desktop did not start"
fi

print -- "Waiting for Docker's engine..."
for ((attempt = 0; attempt < 60; attempt++)); do
    if "$docker_bin" info --format '{{.ServerVersion}}' >/dev/null 2>&1; then
        break
    fi
    sleep 2
done
"$docker_bin" info --format '{{.ServerVersion}}' >/dev/null 2>&1 || fail "Docker's engine is not responding"

print -- "Waiting for Trader Workstation's API. Complete its login if prompted..."
tws_port="${OPT_TRADE_TWS_PORT:-}"
for ((attempt = 0; attempt < 300; attempt++)); do
    if [[ -n "$tws_port" ]]; then
        /usr/bin/nc -z 127.0.0.1 "$tws_port" >/dev/null 2>&1 && break
    elif /usr/bin/nc -z 127.0.0.1 7496 >/dev/null 2>&1; then
        tws_port=7496
        break
    elif /usr/bin/nc -z 127.0.0.1 7497 >/dev/null 2>&1; then
        tws_port=7497
        break
    fi
    sleep 2
done
[[ -n "$tws_port" ]] && /usr/bin/nc -z 127.0.0.1 "$tws_port" >/dev/null 2>&1 || fail "TWS API did not open within 10 minutes. Check API Settings and its socket port."

export OPT_TRADE_TWS_PORT="$tws_port"
if [[ "$tws_port" == 7497 ]]; then
    export OPT_TRADE_IB_GATEWAY_PORT="${OPT_TRADE_IB_GATEWAY_PORT:-4002}"
fi

print -- "Starting Option Stream with TWS port $tws_port..."
cd -- "$project_dir"
"$docker_bin" compose up --build --detach || fail "The Docker container did not start"

for ((attempt = 0; attempt < 30; attempt++)); do
    /usr/bin/nc -z 127.0.0.1 5006 >/dev/null 2>&1 && break
    sleep 2
done
/usr/bin/nc -z 127.0.0.1 5006 >/dev/null 2>&1 || fail "Bokeh did not start on port 5006. Run docker compose logs for details."

/usr/bin/open http://localhost:5006/main || fail "Could not open the browser"
print -- "Option Stream is running at http://localhost:5006/main"
print -- "To stop it later, run: cd '$project_dir' && docker compose down"
