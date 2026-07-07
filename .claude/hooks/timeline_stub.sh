#!/usr/bin/env bash
# timeline_stub.sh — deterministic, append-only staging for the decision timeline.
#
# WHAT IT DOES
#   On session Stop, sweeps several SOURCES for decision artifacts that changed since the
#   last sweep, and appends ONE labelled stub per changed file to docs/decisions/INBOX.md.
#   It NEVER edits docs/DECISION_TIMELINE.md — folding a stub into a real narrative entry
#   (and running plain-speak) stays a deliberate human/LLM step.
#
# WHY A HOOK CAN ONLY DO THIS MUCH
#   A hook runs a shell command; it cannot reason. So it can catch FILE-shaped signals but
#   not conversation-shaped ones. The sources below are the file-shaped feeders:
#     - results/**/*.md            → a verdict/decision file changed          [verdict]
#     - results/section4/cardio_enhance/*.txt → a diagnostic (A/B/C) ran      [diagnostic]
#     - data/vae_v2_diet/*.txt     → a data audit changed                     [data-audit]
#     - tracking.md                → a W&B run was analysed (training-analyst)[wandb-analysis]
#     - **/MASTER_PLAN.md          → a scope was reconciled (sync-plan-tree)  [plan-reconcile]
#   The things it CANNOT see — a decision reasoned out in a Claude chat, a raw W&B run not
#   yet written to tracking.md, an ad-hoc file — are covered by the DELIBERATE path: at the
#   end of a working session, ask Claude to log any gates decided in conversation. The hook
#   guarantees you never forget a file changed; it does not pretend to author the narrative.
#
# SAFE BY DESIGN
#   - append-only to the INBOX, never to the narrative
#   - watermark so only files newer than the last sweep appear (no re-flooding)
#   - hard cap on stubs per sweep, and it says so if it caps (no silent truncation)
#   - always exits 0 so it can never block the session
set -uo pipefail

ROOT="${CLAUDE_PROJECT_DIR:-/home/molefe/PhD/Paper3}"
INBOX="$ROOT/docs/decisions/INBOX.md"
WATERMARK="$ROOT/.claude/hooks/.timeline_watermark"
MAX_STUBS=25

cd "$ROOT" 2>/dev/null || exit 0
[ -f "$INBOX" ]     || exit 0
[ -f "$WATERMARK" ] || touch "$WATERMARK"

# Emit "LABEL<TAB>path" for each changed file, tagged by source so the stub says WHY it
# fired and which capture path applies. Raw prediction CSVs are data, not decisions — skipped.
mapfile -t HITS < <(
  {
    find results -type f -name '*.md' -newer "$WATERMARK" 2>/dev/null \
      | sed 's/^/verdict\t/'
    find results/section4/cardio_enhance -type f -name '*.txt' -newer "$WATERMARK" 2>/dev/null \
      | sed 's/^/diagnostic\t/'
    find data/vae_v2_diet -maxdepth 1 -type f -name '*.txt' -newer "$WATERMARK" 2>/dev/null \
      | sed 's/^/data-audit\t/'
    [ tracking.md -nt "$WATERMARK" ] 2>/dev/null && printf 'wandb-analysis\ttracking.md\n'
    find . -type f -name 'MASTER_PLAN.md' -newer "$WATERMARK" 2>/dev/null \
      | sed 's#^\./##; s/^/plan-reconcile\t/'
  } | grep -v -e '/DECISION_TIMELINE.md' -e '/decisions/' | sort -u
)

# Always advance the watermark so the next sweep only sees genuinely newer files.
touch "$WATERMARK"

[ "${#HITS[@]}" -eq 0 ] && exit 0

TS="$(date '+%Y-%m-%d %H:%M')"
{
  echo ""
  echo "### Sweep $TS — ${#HITS[@]} changed artifact(s)"
  count=0
  for hit in "${HITS[@]}"; do
    label="${hit%%$'\t'*}"
    path="${hit#*$'\t'}"
    if [ "$count" -ge "$MAX_STUBS" ]; then
      echo "- ⚠️ capped at $MAX_STUBS stubs this sweep; $(( ${#HITS[@]} - MAX_STUBS )) more changed — run the sweep again or inspect the sources manually"
      break
    fi
    echo "- [ ] \`[$label]\` \`$path\` changed → log/refresh this gate in DECISION_TIMELINE.md (Question / What we ran / What it said / What we decided / Fired next)"
    count=$(( count + 1 ))
  done
} >> "$INBOX"

exit 0
