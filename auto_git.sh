#!/bin/bash
# Commit and push any changes in this project; run by cron every 8 hours.
cd /home/zcc/claude/cloneSimulation || exit 1
echo "== $(date '+%F %T')"
git add -A
if git diff --cached --quiet; then
  echo "no changes"
else
  git commit -q -m "Auto-commit $(date '+%F %H:%M')" && git push -q origin main && echo "pushed"
fi
