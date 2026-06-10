# 01 · Acquisition + DICOM/Monochrome Decoding

## Reference while you do it
- 📄 Plan: plans/data-foundation/plans/01-acquisition-dicom-monochrome.md

## Section context (paste into the Todoist section)
**Description:** Download/extract the dataset and build a DICOM loader that reads pixel data + tags and normalizes photometric interpretation so intensities are consistent corpus-wide.
**Objective:** Fix the intensity convention once, so no part of the corpus is silently inverted downstream.
**Goal:** A loader turning any DICOM into a normalized "bright = dense" array, verified on a mixed-source panel.
**Verify (whole leaf):** `python -m data.dicom_loader --sample 16 --out figures/monochrome_check.png` → all panels bone-bright/lung-dark regardless of source PhotometricInterpretation.

## Tasks (one at a time)
- [ ] Download and extract the dataset archives to `$HOME/PhD/Paper3/data`; document the disk layout
- [ ] Write a DICOM reader (pydicom) exposing pixel array + tags: PhotometricInterpretation, BitsStored/Allocated, RescaleSlope/Intercept, WindowCenter/Width
- [ ] Invert MONOCHROME1, apply rescale, normalize bit depth to a single convention
- [ ] Render a sample panel mixing MONOCHROME1 and MONOCHROME2 sources to confirm consistent intensity
