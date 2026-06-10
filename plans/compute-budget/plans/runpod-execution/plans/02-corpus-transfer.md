# 📦 Corpus onto the Volume + Integrity Check

## Background
Runs after `01-provision-and-env` — needs the pod up and the repo cloned. The
corpus is the clean NIH ChestX-ray14 set prepared by `data-foundation`
(acquisition + four-group partition), with a manifest of per-file checksums.

## Description
Get the clean corpus onto the pod's network volume and prove it arrived intact.
The default path matches the "git clone + bash" model: re-run the same
download+preprocess bash script on the pod so the corpus is rebuilt on the volume.
A direct transfer (`runpodctl send` / `rsync` from local) is the alternative when
re-downloading is slower than shipping the local copy. Either way, the
post-transfer integrity check is the load-bearing step.

## Purpose
Both trains read from this corpus; a silently truncated or mismatched transfer
would corrupt every checkpoint and metric downstream. Verifying count + checksum
against data-foundation's manifest is what makes "the corpus is on the volume"
trustworthy. Serves Objective 2 and Definition-of-Done #2.

## Goal
The clean corpus on the network volume under a documented path, with a
post-transfer integrity check confirming the image count and per-file checksums
match data-foundation's manifest (zero mismatches).

## Tasks
- [ ] ⚠️ Land the corpus on the volume — default: `bash scripts/download_nih.sh` on the pod; alt: `runpodctl send` / `rsync` the local `data/nih/` to the volume
- [ ] ⚠️ Re-run the four-group partition / preprocess bash step on the pod so the volume holds the same prepared corpus as local
- [ ] ⚠️ Run the integrity check: count images and verify per-file checksums against `data/manifest.parquet` (or `four_group_counts.md`); assert zero mismatches
- [ ] ⚠️ Record the on-volume corpus path + verified counts into `runs/corpus_transfer.md`

## Recommended skill
▶ `/data-integrity-check data/nih` ✅ — count + checksum verification against a manifest is exactly its job.
   — alt: custom shell (`md5sum`/`sha256sum` loop vs the manifest) if the check must run on the bare pod.

## Engagement Instructions
```
# DO THIS — on the pod, rebuild (or receive) the corpus, then verify
$ bash scripts/download_nih.sh /workspace/data/nih      # corpus onto the volume (or: runpodctl send / rsync)
$ bash scripts/verify_corpus.sh /workspace/data/nih data/manifest.parquet

# GET THAT — count + checksums match the manifest, nothing missing or corrupt
$ ls /workspace/data/nih/images | wc -l                 # expect: matches manifest readable count (~112120)
# verify_corpus.sh prints, e.g.:
#   images=112120  checksum_ok=112120  mismatched=0  missing=0   ✅ INTEGRITY OK
$ cat runs/corpus_transfer.md                           # expect: on-volume path + verified counts recorded
```
