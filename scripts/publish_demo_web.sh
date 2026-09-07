#!/usr/bin/env bash
# publish_demo_web.sh — push demo_web/site/ as the root of the GitHub Pages repo.
# Reads repo name / clone dir / Pages URL from configs/demo_web.yaml. Creates the public repo
# and the local clone on first run, enables Pages (main branch, / root) if not yet enabled,
# then rsyncs the built site over the clone, commits and pushes.
#
# Run AFTER scripts/build_demo_web.py:   bash scripts/publish_demo_web.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$REPO_ROOT/configs/demo_web.yaml}"

# pull the three publish settings out of the YAML (one line each, in this order)
read -r GITHUB_REPO REPO_DIR PAGES_URL SITE_DIR < <(cd "$REPO_ROOT" && uv run python -c "
import yaml, os, sys
cfg = yaml.safe_load(open(sys.argv[1]))
p = cfg['publish']
print(p['github_repo'], os.path.expanduser(p['repo_dir']), p['pages_url'], cfg['site_dir'])
" "$CONFIG")
SITE="$REPO_ROOT/$SITE_DIR"
[ -f "$SITE/data.json" ] || { echo "no $SITE/data.json — run scripts/build_demo_web.py first" >&2; exit 1; }

# --- repo + clone (first run only) ---
if ! gh repo view "$GITHUB_REPO" >/dev/null 2>&1; then
  echo "creating public repo $GITHUB_REPO"
  gh repo create "$GITHUB_REPO" --public \
    --description "Listening demo — nine-stem source separation of Gugak ensembles (ISMIR 2026 LBD)"
fi
if [ ! -d "$REPO_DIR/.git" ]; then
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone "git@github.com:$GITHUB_REPO.git" "$REPO_DIR"
  cd "$REPO_DIR"
  git symbolic-ref HEAD refs/heads/main            # make sure the branch is called main
fi

# --- sync the built site over the clone (delete = stale items disappear) ---
# -a archive · --delete remove files not in source · --exclude keep the clone's git metadata
rsync -a --delete --exclude .git "$SITE/" "$REPO_DIR/"
touch "$REPO_DIR/.nojekyll"                        # serve files verbatim, no Jekyll pass

cd "$REPO_DIR"
git add -A
if git diff --cached --quiet; then
  echo "site unchanged — nothing to publish"
else
  NUM_ITEMS=$(python3 -c "import json; d=json.load(open('data.json')); print(sum(len(s['items']) for s in d['sections']))")
  BUILT=$(python3 -c "import json; print(json.load(open('data.json'))['meta']['built_at'])")
  git commit -q -m "[demo] rebuild site: ${NUM_ITEMS} items (built ${BUILT})"
  git push -q -u origin main
  echo "pushed $(git rev-parse --short HEAD) to $GITHUB_REPO"
fi

# --- GitHub Pages: main branch, root — enable once ---
if ! gh api "repos/$GITHUB_REPO/pages" >/dev/null 2>&1; then
  echo "enabling GitHub Pages (main, /)"
  gh api -X POST "repos/$GITHUB_REPO/pages" -f 'source[branch]=main' -f 'source[path]=/' >/dev/null
fi
echo "Pages URL: $PAGES_URL"
