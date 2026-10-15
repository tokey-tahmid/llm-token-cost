#!/bin/bash
# Refuse to commit site-private identifiers into this (publishable) repository.
# Installed as .git/hooks/pre-commit by scripts/install_hooks.sh; also runnable by hand:
#   scripts/check_public.sh            # staged changes
#   scripts/check_public.sh --all      # every tracked file
# The pattern list lives in .private-patterns (git-ignored) so the patterns
# themselves never get published.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PATTERNS=.private-patterns
[[ -s $PATTERNS ]] || { echo "check_public: $PATTERNS missing; refusing to guess" >&2; exit 1; }
if [[ ${1:-} == --all ]]; then
    files=$(git ls-files)
else
    files=$(git diff --cached --name-only --diff-filter=ACM)
fi
[[ -z $files ]] && exit 0
# compressed files are checked by content (packed logs), everything else as text
hits=$(echo "$files" | while IFS= read -r f; do
    if [[ $f == *.gz ]]; then
        gzip -dc -- "$f" | grep -n -i -E -f "$PATTERNS" | sed "s|^|$f:|" || true
    else
        grep -H -n -i -E -I -f "$PATTERNS" -- "$f" 2>/dev/null || true
    fi
done | cut -c1-240)
if [[ -n $hits ]]; then
    echo "check_public: private identifiers found, commit blocked:" >&2
    echo "$hits" | sed 's/^/  /' >&2
    exit 1
fi
exit 0
