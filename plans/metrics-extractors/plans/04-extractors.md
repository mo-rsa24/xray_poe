# 🫀 Heart-Size & Blunting Extractors

## Description
Build the extractors that measure heart size (cardiothoracic-ratio-like) and pleural
blunting / fluid from an image — the features whose *joint* Exp6 tests.

## Purpose
Exp6 is not testable without a reliable extractor of the coupling (heart size and fluid
rising together with severity).

## Goal
Two extractors returning scalar measurements per image, plus a named definition of the coupling.

## Tasks
- [ ] ⚠️ Implement the heart-size extractor
- [ ] ⚠️ Implement the pleural-blunting/fluid extractor
- [ ] ⚠️ Name the coupling the joint must capture (heart size and fluid rising together with severity)

## Engagement Instructions
```
$ python -m metrics.extractors --image <real both-disease>
# expect: prints (heart_size, blunting) scalars in sensible ranges
```
