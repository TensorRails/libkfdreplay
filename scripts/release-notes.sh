#!/usr/bin/env bash
# release-notes.sh TAG [CHANGELOG]: print the changelog section for TAG,
# from "## TAG" up to the next "## ", rewriting relative markdown links to
# point at the tagged tree on GitHub. Fails if the section is missing or empty.
set -euo pipefail
tag=$1
file=${2:-CHANGELOG.md}
notes=$(awk -v tag="$tag" '/^## / { if (found) exit; if ($2 == tag) { found = 1; next } } found' "$file")
if [ -z "${notes//[[:space:]]/}" ]; then
    echo "release-notes: $file has no section for $tag" >&2
    exit 1
fi
base="https://github.com/TensorRails/libkfdreplay/blob/$tag/"
printf '%s\n' "$notes" | sed -E "s#\]\(([^):\#][^):]*)\)#](${base}\1)#g"
