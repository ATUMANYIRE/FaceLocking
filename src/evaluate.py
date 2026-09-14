# src/evaluate.py
"""
Computes genuine vs impostor cosine-distance distributions from enrolled
crops, and suggests an acceptance threshold.

Run:
    python -m src.evaluate
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .embed import ArcFaceEmbedderONNX


ENROLL_DIR = Path("data/enroll")
MIN_IMGS_PER_PERSON = 5
TARGET_FAR = 0.01  # aim for at most 1% false-accept rate


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(np.dot(a.reshape(-1), b.reshape(-1)))


def load_embeddings_for_person(embedder: ArcFaceEmbedderONNX, person_dir: Path) -> List[np.ndarray]:
    embs = []
    for img_path in sorted(person_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        res = embedder.embed(img)
        embs.append(res.embedding)
    return embs


def pairwise_distances(embs_a: List[np.ndarray], embs_b: List[np.ndarray], same: bool) -> List[float]:
    dists = []
    if same:
        for i in range(len(embs_a)):
            for j in range(i + 1, len(embs_a)):
                dists.append(cosine_distance(embs_a[i], embs_a[j]))
    else:
        for ea in embs_a:
            for eb in embs_b:
                dists.append(cosine_distance(ea, eb))
    return dists


def describe(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "n=0"
    return (
        f"n={arr.size} mean={arr.mean():.3f} std={arr.std():.3f} "
        f"p05={np.percentile(arr,5):.3f} p50={np.percentile(arr,50):.3f} p95={np.percentile(arr,95):.3f}"
    )


def main():
    embedder = ArcFaceEmbedderONNX(debug=False)

    people_dirs = sorted([p for p in ENROLL_DIR.iterdir() if p.is_dir()])
    per_person: Dict[str, List[np.ndarray]] = {}

    for pdir in people_dirs:
        embs = load_embeddings_for_person(embedder, pdir)
        if len(embs) >= MIN_IMGS_PER_PERSON:
            per_person[pdir.name] = embs
            print(f"Loaded {len(embs)} crops for {pdir.name}")
        else:
            print(f"Skipping {pdir.name}: only {len(embs)} crops (need {MIN_IMGS_PER_PERSON}+)")

    names = sorted(per_person.keys())
    if len(names) < 2:
        print("\nNeed at least 2 enrolled people to compute impostor distances. Enroll another person.")
        return

    genuine_all: List[float] = []
    for name in names:
        genuine_all.extend(pairwise_distances(per_person[name], per_person[name], same=True))

    impostor_all: List[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            impostor_all.extend(pairwise_distances(per_person[names[i]], per_person[names[j]], same=False))

    genuine = np.array(genuine_all, dtype=np.float32)
    impostor = np.array(impostor_all, dtype=np.float32)

    print("\n=== Distance Distributions ===")
    print(f"Genuine (same person):   {describe(genuine)}")
    print(f"Impostor (diff people):  {describe(impostor)}")

    thresholds = np.arange(0.10, 1.20, 0.02)
    best = None
    print("\n=== Threshold Sweep ===")
    for thr in thresholds:
        far = float(np.mean(impostor <= thr)) if impostor.size else 0.0
        frr = float(np.mean(genuine > thr)) if genuine.size else 0.0
        print(f"thr={thr:.2f}  FAR={far*100:5.2f}%  FRR={frr*100:5.2f}%")
        if far <= TARGET_FAR and (best is None or frr < best[2]):
            best = (thr, far, frr)

    if best:
        thr, far, frr = best
        print(f"\nSuggested threshold: {thr:.2f}  (FAR={far*100:.2f}%, FRR={frr*100:.2f}%)")
    else:
        print("\nNo threshold met the target FAR. Your two identities may be too visually similar,")
        print("or you need more/varied samples. Consider using the smallest-FAR threshold above.")


if __name__ == "__main__":
    main()