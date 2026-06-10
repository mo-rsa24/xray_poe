# 📈 Single-Disease FID Gate

## Description
Generate single-disease samples and measure FID against real single-disease images;
confirm both the `∅` and `normal` nulls are callable at inference.

## Purpose
Unconvincing single-disease samples make the composition test meaningless — this gate
guards Exp5/6.

## Goal
FID reported (passing gate), a sample grid saved, both anchors confirmed callable.

## Tasks
- [ ] ⚠️ Sample per single-disease condition; compute FID vs real single-disease
- [ ] ⚠️ Save a sample grid; confirm sampling with null=`∅` and null=`normal` both work

## Engagement Instructions
```
$ python -m ldm.evaluate --ckpt ckpts/ldm_*.pt --fid
# expect: FID <= gate; figures/ldm_samples.png; both null modes produce samples
```
