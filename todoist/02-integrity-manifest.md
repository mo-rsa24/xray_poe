# 02 · Integrity Scan + Manifest

## Reference while you do it
- 📄 Plan: plans/data-foundation/plans/02-integrity-manifest.md

## Section context (paste into the Todoist section)
**Description:** Scan every DICOM for readability, quarantine corrupt/truncated files, and emit a manifest linking each clean image to its labels and key tags.
**Objective:** Produce the single source of truth every later scope joins against.
**Goal:** A parquet manifest (path, labels, photometric, resolution, bit depth) + a quarantine log.
**Verify (whole leaf):** `python -m data.build_manifest --root data/ --out data/manifest.parquet` → prints `readable=N, quarantined=M` (M small).

## Tasks (one at a time)
- [ ] Iterate all DICOMs; attempt decode; record failures with reason; quarantine corrupt/truncated
- [ ] Join images to the label table; emit manifest (path → label(s) + tags)
- [ ] Report counts: total, readable, quarantined
