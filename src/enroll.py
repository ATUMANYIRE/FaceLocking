# src/enroll.py
"""
Enrollment: capture face samples for a person, average into one template
embedding, save to the database.

Run:
    python -m src.enroll

Controls:
    SPACE : capture one sample
    a     : toggle auto-capture (captures every ~0.25s)
    s     : save enrollment (needs a handful of samples first)
    r     : reset newly-captured samples (keeps anything already saved to disk)
    q     : quit
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .haar_5pt import Haar5ptDetector, align_face_5pt
from .embed import ArcFaceEmbedderONNX


@dataclass
class EnrollConfig:
    out_db_npz: Path = Path("data/db/face_db.npz")
    out_db_json: Path = Path("data/db/face_db.json")
    crops_dir: Path = Path("data/enroll")
    samples_needed: int = 15
    auto_capture_every_s: float = 0.25


def ensure_dirs(cfg: EnrollConfig) -> None:
    cfg.out_db_npz.parent.mkdir(parents=True, exist_ok=True)
    cfg.crops_dir.mkdir(parents=True, exist_ok=True)


def load_db(cfg: EnrollConfig) -> Dict[str, np.ndarray]:
    if cfg.out_db_npz.exists():
        data = np.load(cfg.out_db_npz, allow_pickle=True)
        return {k: data[k].astype(np.float32) for k in data.files}
    return {}


def save_db(cfg: EnrollConfig, db: Dict[str, np.ndarray], meta: dict) -> None:
    ensure_dirs(cfg)
    np.savez(cfg.out_db_npz, **{k: v.astype(np.float32) for k, v in db.items()})
    cfg.out_db_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def mean_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    """Average several embeddings, then re-normalize to unit length."""
    E = np.stack([e.reshape(-1) for e in embeddings], axis=0).astype(np.float32)
    m = E.mean(axis=0)
    m = m / (np.linalg.norm(m) + 1e-12)
    return m.astype(np.float32)


def main():
    cfg = EnrollConfig()
    ensure_dirs(cfg)

    name = input("Enter person name to enroll (e.g., Alice): ").strip()
    if not name:
        print("No name provided. Exiting.")
        return

    det = Haar5ptDetector(smooth_alpha=0.80, debug=False)
    emb = ArcFaceEmbedderONNX(debug=False)

    db = load_db(cfg)
    person_dir = cfg.crops_dir / name
    person_dir.mkdir(parents=True, exist_ok=True)

    new_samples: List[np.ndarray] = []
    status_msg = ""
    auto = False
    last_auto = 0.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened.")

    time.sleep(0.5)
    for _ in range(10):
        cap.read()

    print("\nEnrollment started.")
    print("SPACE=capture, a=auto-capture, s=save, r=reset, q=quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        vis = frame.copy()
        faces = det.detect(frame, max_faces=1)
        aligned: Optional[np.ndarray] = None

        if faces:
            f = faces[0]
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            cv2.imshow("aligned_112", aligned)

        now = time.time()
        if auto and aligned is not None and (now - last_auto) >= cfg.auto_capture_every_s:
            r = emb.embed(aligned)
            new_samples.append(r.embedding)
            last_auto = now
            status_msg = f"Auto captured ({len(new_samples)})"
            fn = person_dir / f"{int(now * 1000)}.jpg"
            cv2.imwrite(str(fn), aligned)

        lines = [
            status_msg,
            f"ENROLL: {name}",
            f"Samples: {len(new_samples)} / {cfg.samples_needed}",
            f"Auto: {'ON' if auto else 'OFF'}",
            "SPACE=capture | s=save | r=reset | q=quit",
        ]
        y = 30
        for line in lines:
            if line:
                cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25

        cv2.imshow("enroll", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("a"):
            auto = not auto
        if key == ord("r"):
            new_samples.clear()
            status_msg = "Reset."
        if key == ord(" "):
            if aligned is None:
                status_msg = "No face detected."
            else:
                r = emb.embed(aligned)
                new_samples.append(r.embedding)
                status_msg = f"Captured ({len(new_samples)})"
                fn = person_dir / f"{int(time.time() * 1000)}.jpg"
                cv2.imwrite(str(fn), aligned)
        if key == ord("s"):
            if len(new_samples) < 5:
                status_msg = f"Need more samples (have {len(new_samples)}, want 5+)."
                continue

            template = mean_embedding(new_samples)
            db[name] = template
            meta = {
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "embedding_dim": int(template.size),
                "names": sorted(db.keys()),
                "samples_used": int(len(new_samples)),
            }
            save_db(cfg, db, meta)
            status_msg = f"Saved '{name}'. Total identities: {len(db)}"
            print(status_msg)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()