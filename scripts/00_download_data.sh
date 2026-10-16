#!/bin/bash
# Download the public datasets used by model v1 into data/ (paths and URLs come from configs/v1.yaml).
#
#   bash scripts/00_download_data.sh                     # everything (~35 GB: H1 ~34 GB, Replogle ~0.55 GB, GENCODE 30 MB)
#   bash scripts/00_download_data.sh replogle gencode    # only some groups: replogle | gencode | h1
#   bash scripts/00_download_data.sh --list              # show the plan and what is already on disk (no network)
#   bash scripts/00_download_data.sh --config other.yaml
#
# Safe to re-run: complete files are skipped (size compared with the server's), interrupted downloads resume
# (curl -C -) from a .part file, and a file only gets its final name once it is complete.
#
# On TSUBAME: download ONCE for the whole team on the free interactive queue (iqrsh), into the group disk, and link
# it into each clone, so paid jobs never spend time downloading:
#   mkdir -p /gs/bs/<group>/shared/data && ln -s /gs/bs/<group>/shared/data data
#   bash scripts/00_download_data.sh
set -euo pipefail

cd "$(dirname "$0")/.."                       # project root, whatever the current directory is
CONFIG=configs/v1.yaml
LIST_ONLY=0
GROUPS_WANTED=""
while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --list) LIST_ONLY=1; shift ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        replogle|gencode|h1) GROUPS_WANTED="$GROUPS_WANTED $1"; shift ;;
        *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done
[ -n "$GROUPS_WANTED" ] || GROUPS_WANTED="replogle gencode h1"

PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3                     # needs PyYAML

# The config → one line per file: group <TAB> local path <TAB> expected bytes or "-" <TAB> URLs (space-separated).
PLAN=$("$PY" - "$CONFIG" <<'EOF'
import sys
from pathlib import Path
import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
data_dir = Path(cfg["paths"]["data_dir"])
d = cfg["data"]
for name, r in d["replogle"].items():
    urls = " ".join(f"{base.rstrip('/')}/{r['figshare_id']}" for base in d["figshare_urls"])
    print(f"replogle\t{data_dir / r['file']}\t-\t{urls}")
print(f"gencode\t{data_dir / d['gencode']['file']}\t-\t{d['gencode']['url']}")
h1 = d["h1"]
for remote, size in h1["files"].items():
    print(f"h1\t{data_dir / h1['dir'] / Path(remote).name}\t{size if size else '-'}\t{h1['base_url'].rstrip('/')}/{remote}")
EOF
)

file_size() { if [ -f "$1" ]; then wc -c < "$1" | tr -d ' '; else echo 0; fi; }
human() { awk -v b="$1" 'BEGIN { split("B KB MB GB TB", u); i = 1; while (b >= 1000 && i < 5) { b /= 1000; i++ }; printf "%.1f %s", b, u[i] }'; }

# Total size of a remote file from a 1-byte range request (Content-Range: bytes 0-0/TOTAL). A HEAD request is not
# enough: figshare redirects to signed storage URLs that only accept GET.
remote_size() {
    curl -sL --max-time 120 -r 0-0 -D - -o /dev/null "$1" 2>/dev/null | tr -d '\r' |
        awk 'tolower($1) == "content-range:" { split($3, a, "/"); n = a[2] } END { if (n ~ /^[0-9]+$/) print n }'
}

echo "Config: $CONFIG    groups: $GROUPS_WANTED"
DATA_DIR=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['paths']['data_dir'])" "$CONFIG")
mkdir -p "$DATA_DIR"
echo "Free space where data goes:"; df -h "$DATA_DIR" | tail -1
echo "Note: the three H1 files total ~34 GB."
echo

STATUS=0
while IFS="$(printf '\t')" read -r group dest expected urls; do
    case " $GROUPS_WANTED " in *" $group "*) ;; *) continue ;; esac
    have=$(file_size "$dest")
    if [ "$LIST_ONLY" = 1 ]; then
        exp_txt="unknown size"; [ "$expected" != "-" ] && exp_txt=$(human "$expected")
        state="missing"; [ "$have" -gt 0 ] && state="on disk: $(human "$have")"
        [ "$expected" != "-" ] && [ "$have" = "$expected" ] && state="complete"
        printf '  %-9s %-55s %-14s %s\n' "$group" "$dest" "($exp_txt)" "$state"
        continue
    fi
    set -- $urls
    expected_bytes="$expected"
    [ "$expected_bytes" = "-" ] && expected_bytes=$(remote_size "$1" || true)
    if [ "$have" -gt 0 ] && [ -n "$expected_bytes" ] && [ "$have" = "$expected_bytes" ]; then
        echo "  $dest: complete ($(human "$have")), skipping"
        continue
    fi
    if [ "$have" -gt 0 ] && [ -z "$expected_bytes" ]; then
        echo "  $dest: exists ($(human "$have")) and the server did not report a size; skipping (delete it to re-download)"
        continue
    fi
    mkdir -p "$(dirname "$dest")"
    # A partial file under the final name (e.g. from an older script) becomes the .part file we resume.
    [ "$have" -gt 0 ] && mv "$dest" "$dest.part"
    echo "  $dest: downloading ($([ -n "$expected_bytes" ] && human "$expected_bytes" || echo "size unknown")) …"
    ok=0
    for url in "$@"; do
        if curl -fL --retry 5 --retry-delay 10 --connect-timeout 30 -C - -o "$dest.part" "$url"; then ok=1; break; fi
        echo "    failed from $url" >&2
    done
    got=$(file_size "$dest.part")
    if [ "$ok" = 1 ] && { [ -z "$expected_bytes" ] || [ "$got" = "$expected_bytes" ]; }; then
        mv "$dest.part" "$dest"
        echo "    done: $(human "$got")"
    else
        echo "    INCOMPLETE: $dest.part has $(human "$got") (expected ${expected_bytes:-?} bytes); re-run to resume" >&2
        STATUS=1
    fi
done <<EOF
$PLAN
EOF

if [ "$LIST_ONLY" = 0 ]; then
    echo; echo "Sizes on disk:"
    du -sh "$DATA_DIR"/* 2>/dev/null || true
fi
exit $STATUS
