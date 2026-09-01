#!/usr/bin/env bash

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

entry="${1:-}"
entry="${entry%.py}"
if [ -z "$entry" ] || [ ! -f "$root/$entry.py" ]; then
    echo "usage: ./run.sh <entry> [args...]"
    echo "entry: $(cd "$root" && ls *.py probe/*.py | sed 's/\.py$//' | tr '\n' ' ')"
    exit 1
fi
shift

session="$(basename "$root")"
remote="/content/$session"
tmp="$(mktemp -t "$session")"

model=""
tag=""
args=()
all=("$@")
i=0
while [ $i -lt ${#all[@]} ]; do
    a="${all[i]}"
    if [ "$a" = "--tag" ]; then
        tag="${all[i + 1]:-}"
        i=$((i + 2))
        continue
    fi
    if [ "$a" = "--model" ]; then
        model="${all[i + 1]:-}"
    fi
    args+=("$a")
    i=$((i + 1))
done
set -- ${args[@]+"${args[@]}"}
if [ -z "$model" ]; then
    model="$(sed -n 's/^ *model: *str *= *"\(.*\)".*/\1/p' "$root/$entry.py" | head -1)"
fi
log="logs/$(basename "${model:-default}")_${entry//\//_}${tag:+_$tag}.log"

mkdir -p "$root/logs"
colab new -s "$session" --gpu A100
trap 'rm -f "$tmp"; colab stop -s "$session" >/dev/null 2>&1' EXIT

colab install -s "$session" \
    accelerate \
    datasets \
    transformers \
    simple-parsing \
    hqq pymoo scipy

tar czf "$tmp" -C "$root" --no-xattrs --no-mac-metadata \
    --exclude '__pycache__' \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude 'logs' \
    --exclude '.DS_Store' .
colab upload -s "$session" "$tmp" "$remote.tgz"

colab exec -s "$session" --timeout 300 <<EOF
import subprocess
subprocess.run("mkdir -p $remote && tar xzf $remote.tgz -C $remote && rm $remote.tgz", shell=True, check=True)
EOF

echo "$log"

colab exec -s "$session" --timeout 86400 <<EOF > "$root/$log"
import subprocess
proc = subprocess.Popen(
    ["bash", "-c", "cd $remote && set -a && [ -f .env ] && . ./.env; set +a; TQDM_MININTERVAL=30 PYTHONPATH=. python -u $entry.py $* 2>&1"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
for line in proc.stdout:
    if line.strip():
        print(line, end="", flush=True)
if proc.wait():
    raise SystemExit(proc.returncode)
EOF
