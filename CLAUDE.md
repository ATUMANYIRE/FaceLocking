# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Real-time face recognition (Haar detection → MediaPipe 5-point landmarks → ArcFace alignment → ONNX embedding → cosine matching), with optional pan-servo face tracking via an ESP8266 over serial. Windows-oriented (default serial port `COM5`). See README.md for the full stage-by-stage description and keyboard controls.

## Commands

There is no build step, linter config, or test suite. Every module in `src/` is a standalone interactive OpenCV script run as a package module **from the repo root** (all model/data paths are relative to the root, and modules use relative imports, so `python src/foo.py` will fail):

```bash
pip install opencv-python mediapipe onnxruntime numpy pyserial   # a .venv/ exists locally
python -m src.embed            # sanity-check the ONNX model: [1,3,112,112] -> [1,512]
python -m src.enroll           # prompts for a name, captures samples, writes the DB
python -m src.recognize        # live recognition
python -m src.recognize_track  # recognition + servo search/lock tracking
python -m src.face_lock --name ATUMANYIRE [--camera 1]  # multi-face labels, lock, nose position, expressions, blinks; no servo
python -m src.evaluate         # genuine/impostor distance sweep to pick a threshold
python -m mqtt_tracking.face_track_mqtt [--host IP] [--mode fixed|onboard] [--invert] [--width 1280 --height 720]  # tracking over MQTT -> ESP8266 stepper
python -m mqtt_tracking.mqtt_publisher  # camera-free MQTT smoke test (needs a broker)
```

All scripts need a webcam (index 0) and open GUI windows; they cannot be verified headlessly. `models/embedder_arcface.onnx` (~166 MB) is git-ignored and must be supplied manually — anything past landmarks fails without it.

## Architecture

- **`src/haar_5pt.py` is the shared core.** `Haar5ptDetector.detect()` and `align_face_5pt()` are used by every later stage (embed, enroll, recognize, recognize_track). Changes here affect the whole pipeline.
  - Haar only *gates* detection: the largest Haar box is used to validate FaceMesh points (≥60% inside a 35%-padded box, plus eye-distance / mouth-below-nose sanity checks). The returned bounding box is rebuilt from the 5 keypoints, not taken from Haar.
  - FaceMesh defaults to `max_mesh_faces=1`, so every script except `face_lock.py` is effectively single-face.
  - The detector is **stateful**: box and keypoints are EMA-smoothed across frames (`smooth_alpha`, default 0.80). Reuse one instance per video stream; don't share it across unrelated images.
  - Keypoint order is fixed as `[left_eye, right_eye, nose, mouth_left, mouth_right]` and must match the ArcFace template in `_estimate_norm_5pt`.
- **`src/embed.py`** — `ArcFaceEmbedderONNX` takes an aligned 112×112 **BGR** crop, converts to RGB, normalizes `(x-127.5)/128`, returns an L2-normalized 512-D vector (CPU provider only).
- **Face DB** — `data/db/face_db.npz` stores one mean, re-normalized template per name (key = name); `face_db.json` is metadata only. Re-enrolling a name overwrites its template. Aligned crops are saved to `data/enroll/<NAME>/*.jpg`; `evaluate.py` re-embeds those crops directly (no re-alignment), so they must remain aligned 112×112 images. Everything generated under `data/` (db, enroll, debug_aligned, action_history.jsonl) is git-ignored.
- **Matching** — distance = `1 - cosine_similarity`; accept if `≤ DEFAULT_THRESHOLD` (0.34). `recognize.py` and `recognize_track.py` each define their own `load_db`/`Matcher`/threshold constant (duplicated, not shared) — keep them in sync when changing matching logic.
- **Tracking** (`recognize_track.py`) — state machine: sweep (`Sweeper`, 30–150°) until an accepted face appears → lock onto that name → follow its horizontal offset (`angle = 90 - norm*60`, only sent if change ≥ `MIN_ANGLE_DELTA`) → release after `LOCK_LOSS_FRAMES` misses and resume sweep from the last angle. A lock only persists if the same name is re-matched within `LOCK_MATCH_RADIUS` px.
- **Face lock + expressions** (`face_lock.py`, `expression.py`) — no servo; reuses `Matcher`/`load_db` from `recognize.py`. Uses `Haar5ptDetector(max_mesh_faces=N).detect_all()`, which returns every FaceMesh face confirmed by its own Haar box, unsmoothed (`detect()` stays single-face + EMA for the other scripts). Every face is embedded and labelled; only the target can acquire the lock, after `LOCK_CONFIRM_FRAMES` consecutive matches. Lock constants (`LOCK_MATCH_RADIUS` = 150 here vs 120 in `recognize_track.py`) are defined separately per script. Lock acquired/released, movement, expression-change and blink events are appended as JSONL to `data/action_history.jsonl` (`--history` overrides). The lock follows the face recognized as the locked name, else the face nearest the last nose position, so it survives recognition dropouts during big expressions; it releases on `IMPOSTOR_DIST` / other-name matches for `LOCK_SWITCH_FRAMES`, or after `LOCK_LOSS_FRAMES` missing. Position is the nose tip (`kps[2]`) vs frame centre in unmirrored screen coordinates. `ExpressionDetector` uses the full FaceMesh (`FaceKpsBox.mesh`), rotates it upright, normalizes by eye-corner distance, and classifies by *delta from a per-lock neutral baseline* (first 30 frames); blinks use the raw eye aspect ratio vs the calibrated open-eye EAR. All thresholds are rule-based constants at the top of `expression.py`.
- **Servo protocol** (`servo_control.py`) — writes `"<angle>\n"` as ASCII at 9600 baud. Opening the port resets the ESP8266 via DTR, so the controller waits `BOOT_SETTLE_S` (4 s) and retries reconnects on write failure; `pyserial` is an optional import. The firmware side is `sketch_sep12a.ino` (repo root): reads with `Serial.parseInt()`, ignores angles outside 0–180, drives the servo on GPIO2 (board label D4), and centres at 90° on boot.
- **MQTT stepper variant** (`mqtt_tracking/`, see its README) — `face_track_mqtt.py` reuses `Matcher`/`Sweeper`/constants from `recognize_track.py`, but publishes through `MqttAngleController` (same `set_angle` interface as `ServoController`, paho async + auto-reconnect). Topics under `face_tracking/`: `angle` (ASCII 0–180), `cmd` (`home`/`release`), `status`/`position` from the device. The default `fixed` mode (stationary laptop webcam) uses the absolute servo formula; `onboard` is closed-loop (camera on the motor; the target is nudged by `TRACK_GAIN_DEG` every `TRACK_UPDATE_S`). Firmware `esp8266_stepper_mqtt.ino`: PubSubClient, 28BYJ-48 half-step (4096 steps/rev) through a ULN2003 on D1/D2/D5/D6, non-blocking stepping, and the coils switch off when idle. There is no position sensor, so the power-on position is taken as 90°. Mosquitto must listen on `0.0.0.0` (see `mqtt_tracking/mosquitto.conf`) for the ESP8266 to connect.

## Notes

- Enrollment: `EnrollConfig.samples_needed` is 15 (target shown in the UI), but `s` saves with as few as 5 samples.
- `DEFAULT_THRESHOLD` comments say it came from `evaluate.py`; rerun evaluation after enrolling new people or changing the model/alignment.
