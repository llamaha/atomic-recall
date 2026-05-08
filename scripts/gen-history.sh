#!/usr/bin/env bash
# gen-history.sh — Replay git log as Atomic changes for testing atomic-recall
#
# Usage: ./scripts/gen-history.sh [git-repo] [max-commits]
#   git-repo     Path to a git repository to replay history from (default: /tmp/ripgrep-git)
#   max-commits  Maximum number of commits to replay (default: 500)

set -euo pipefail

GIT_REPO="${1:-/tmp/ripgrep-git}"
MAX_COMMITS="${2:-500}"
ATOMIC_REPO="$(pwd)"
SCRATCH="$ATOMIC_REPO/.gen-history-scratch"

if [ ! -d "$GIT_REPO/.git" ]; then
    echo "Error: $GIT_REPO is not a git repository"
    exit 1
fi

if [ ! -d "$ATOMIC_REPO/.atomic" ]; then
    echo "Error: run this from an Atomic repository root"
    exit 1
fi

echo "Source git repo:  $(realpath $GIT_REPO)"
echo "Max commits:      $MAX_COMMITS"
echo "Target atomic repo: $ATOMIC_REPO"
echo ""

# Seed the scratch file and record it so atomic has a clean baseline
echo "init" > "$SCRATCH"
atomic add "$SCRATCH" 2>/dev/null || true
atomic record -m "gen-history: initialize scratch file" 2>/dev/null || true

# Read commits oldest-first: hash, author name, author email, subject
commits=$(git -C "$GIT_REPO" log \
    --reverse \
    --max-count="$MAX_COMMITS" \
    --format="%H%x09%an%x09%ae%x09%s")

total=$(echo "$commits" | grep -c . || true)
echo "Replaying $total commits..."
echo ""

count=0
recorded=0
skipped=0

while IFS=$'\t' read -r hash author_name author_email subject; do
    [ -z "$hash" ] && continue
    count=$((count + 1))

    # Always change the file so atomic detects a diff
    printf '%s\n%s\n' "$count" "$hash" > "$SCRATCH"

    # Sanitise the subject — strip chars that break shell quoting
    safe_subject=$(printf '%s' "$subject" | tr -d '\000-\037' | cut -c1-200)
    [ -z "$safe_subject" ] && safe_subject="(no message)"

    if atomic record \
        --author "$author_name <$author_email>" \
        -m "$safe_subject" 2>/dev/null; then
        recorded=$((recorded + 1))
        printf "\r  [%d/%d] %s" "$count" "$total" "${safe_subject:0:65}"
    else
        skipped=$((skipped + 1))
        printf "\r  [%d/%d] SKIPPED" "$count" "$total"
    fi

done <<< "$commits"

echo ""
echo ""
echo "Done. $recorded recorded, $skipped skipped."
echo ""
echo "Next steps:"
echo "  atomic log                                          — verify history"
echo "  atomic-recall index --repo .                       — index all changes"
echo "  atomic-recall search --repo . \"your query\"         — search"
