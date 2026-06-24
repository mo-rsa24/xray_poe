"""Run per-class FID (inception + xrv) for the pilot ladder and merge to one JSON.

Real references are class-matched (images_cardio_only / images_effusion_only) rather
than the plan's data/nih/images/, which now holds the full 112k corpus and would
conflate pathologies. A real-vs-real split floor is included per (class, embed).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics.fid import extract_features, fid_from_features

N = 200
DEVICE = "cuda"
CLASSES = {
    "cardiomegaly": ("data/nih/images_cardio_only", "outputs/ckpt_eval/0100000/single/cardiomegaly"),
    "effusion": ("data/nih/images_effusion_only", "outputs/ckpt_eval/0100000/single/effusion"),
}
EMBEDS = ["inception", "xrv"]


def gather(d, n):
    ps = sorted(Path(d).glob("*.png")) + sorted(Path(d).glob("*.jpg"))
    return ps[:n]


out = {"n_per_set": N, "real_ref": "class-matched (images_cardio_only / images_effusion_only)",
       "note": "plan said data/nih/images/ but that is now the full corpus; class-matched is the honest per-class ref",
       "classes": {}}

for cls, (real_dir, gen_dir) in CLASSES.items():
    real_paths = gather(real_dir, N)
    gen_paths = gather(gen_dir, N)
    half = len(real_paths) // 2
    out["classes"][cls] = {"real_dir": real_dir, "gen_dir": gen_dir,
                           "n_real": len(real_paths), "n_gen": len(gen_paths)}
    for embed in EMBEDS:
        fr = extract_features(real_paths, embed=embed, device=DEVICE)
        fg = extract_features(gen_paths, embed=embed, device=DEVICE)
        fid_gen = float(fid_from_features(fr, fg)["fid"])
        fid_floor = float(fid_from_features(fr[:half], fr[half:])["fid"])
        out["classes"][cls][embed] = {"fid_gen_vs_real": fid_gen,
                                      "fid_real_vs_real_floor": fid_floor}
        print(f"{cls:13s} {embed:10s} FID(gen,real)={fid_gen:8.2f}  floor(real,real)={fid_floor:8.2f}")

Path("results").mkdir(exist_ok=True)
Path("results/pilot_fid.json").write_text(json.dumps(out, indent=2))
print("\nResults → results/pilot_fid.json")
