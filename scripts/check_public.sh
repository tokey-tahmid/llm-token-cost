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
hits=$(echo "$files" | xargs -I{} grep -H -n -i -E -f "$PATTERNS" -- {} 2>/dev/null || true)
if [[ -n $hits ]]; then
    echo "check_public: private identifiers found, commit blocked:" >&2
    echo "$hits" | sed 's/^/  /' >&2
    exit 1
fi
exit 0
