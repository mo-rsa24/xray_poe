# 📥 Acquisition + DICOM/Monochrome Decoding

## Description
Download and extract the chest X-ray dataset, build a DICOM loader that reads pixel
data and the load-bearing tags, and normalize photometric interpretation so pixel
intensities are consistent across the whole corpus.

## Purpose
Without correct monochrome handling, part of the corpus can be intensity-inverted,
silently corrupting every downstream model and metric. This fixes the convention once.

## Goal
A loader that turns any DICOM in the dataset into a normalized array with a consistent
"bright = dense" convention, verified on a mixed-source sample panel.

## Tasks
- [ ] ⚠️ Download and extract the dataset archives to `$HOME/PhD/Paper3/data`; document the disk layout
- [ ] ⚠️ Write a DICOM reader (pydicom) exposing pixel array + tags: PhotometricInterpretation, BitsStored/Allocated, RescaleSlope/Intercept, WindowCenter/Width
- [ ] ⚠️ Invert MONOCHROME1, apply rescale, normalize bit depth to a single convention
- [ ] ⚠️ Render a sample panel mixing MONOCHROME1 and MONOCHROME2 sources to confirm consistent intensity

## Engagement Instructions
```
$ python -m data.dicom_loader --sample 16 --out figures/monochrome_check.png
# expect: all 16 panels show bone bright / lung dark, regardless of source PhotometricInterpretation
$ python -c "from data.dicom_loader import load; a=load('<a MONOCHROME1 file>'); print(a.min(), a.max(), a.dtype)"
# expect: normalized range, consistent dtype
```
