# 02 · Corpus onto the volume + integrity check

[⌂ Index](00-INDEX.md) · [← prev 01](01-provision-and-env.md) · [next → 03](03-vae-train-on-pod.md)

## Reference while you do it
- 📄 Plan: plans/compute-budget/plans/runpod-execution/plans/02-corpus-transfer.md

## Section context (paste into the Todoist section)
**Description:** Get the clean NIH corpus onto the pod's network volume and prove it arrived intact. Default path re-runs the download+preprocess bash script on the pod; alt is a direct transfer from local. The post-transfer integrity check is the load-bearing step.
**Objective:** Make the corpus both trains read from trustworthy — a silently truncated transfer would corrupt every checkpoint and metric downstream.
**Goal:** The clean corpus on the volume under a documented path, with a post-transfer integrity check confirming image count and per-file checksums match data-foundation's manifest (zero mismatches).
**Verify (whole leaf):** `bash scripts/verify_corpus.sh /workspace/data/nih data/manifest.parquet` → `images=… checksum_ok=… mismatched=0 missing=0 ✅ INTEGRITY OK`; `ls /workspace/data/nih/images | wc -l` matches the manifest readable count (~112120); `cat runs/corpus_transfer.md` records the path + verified counts.
**▶ Recommended prompt:** `/data-integrity-check data/nih` ✅ — count + checksum verification against a manifest. alt: custom shell (`md5sum`/`sha256sum` loop vs the manifest).

## Tasks (one at a time)
- [ ] Land the corpus on the volume — default: `bash scripts/download_nih.sh` on the pod; alt: `runpodctl send` / `rsync` the local `data/nih/` to the volume
- [ ] Re-run the four-group partition / preprocess bash step on the pod so the volume holds the same prepared corpus as local
- [ ] Run the integrity check: count images and verify per-file checksums against `data/manifest.parquet` (or `four_group_counts.md`); assert zero mismatches
- [ ] Record the on-volume corpus path + verified counts into `runs/corpus_transfer.md`
