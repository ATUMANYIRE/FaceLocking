# src/align.py
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .haar_5pt import Haar5ptDetector, align_face_5pt


def main(cam_index: int = 0, out_size: Tuple[int, int] = (112, 112)):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened.")

    time.sleep(0.5)
    for _ in range(10):
        cap.read()

    det = Haar5ptDetector(smooth_alpha=0.80, debug=True)

    out_w, out_h = out_size
    last_aligned = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    save_dir = Path("data/debug_aligned")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("align running. Press 'q' to quit, 's' to save aligned face.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        faces = det.detect(frame, max_faces=1)
        vis = frame.copy()

        if faces:
            f = faces[0]
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            for (x, y) in f.kps.astype(int):
                cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)

            aligned, _ = align_face_5pt(frame, f.kps, out_size=out_size)
            if aligned.size:
                last_aligned = aligned
            cv2.putText(vis, "OK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("align - camera", vis)
        cv2.imshow("align - aligned", last_aligned)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            ts = int(time.time() * 1000)
            out_path = save_dir / f"{ts}.jpg"
            cv2.imwrite(str(out_path), last_aligned)
            print(f"[align] saved: {out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()