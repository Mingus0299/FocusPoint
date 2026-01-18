# Polygon Tracking System — How It Works

This document explains the tracking algorithm used in our demo:
what algorithms are involved, how the pipeline works step by step, and why we chose a classical computer vision approach instead of deep learning.

---

## One-Paragraph Pitch (TL;DR)

We use a classical computer vision pipeline optimized for noisy ultrasound video. The user initializes a polygon, and we track strong feature points along its boundary using pyramidal Lucas–Kanade optical flow with forward–backward validation. Each frame, we robustly estimate an affine motion model using RANSAC and refine it with least-squares for precision. To handle ultrasound noise and sudden motion, we apply confidence-based filtering, smoothing, and a two-stage template recovery mechanism. This allows the annotation to follow translation, rotation, and scale changes in real time without requiring any training data.

---

## Why Not Deep Learning?

We intentionally avoided deep learning for three reasons:

1. **Data availability**Medical ultrasound requires large, carefully labeled datasets that are difficult and expensive to obtain. Models trained on one dataset often do not generalize well across machines, hospitals, or imaging protocols.
2. **Interpretability and reliability**Classical methods give us full visibility into *why* tracking succeeds or fails. We can expose confidence, recovery events, and failure modes in real time, which is critical for medical workflows.
3. **Hackathon constraints**
   Our approach runs in real time on CPU, requires no training phase, and can be deployed immediately. This makes it ideal for a rapid prototype while still being clinically meaningful.

That said, this system is designed to be **ML-ready**: a learned segmentation or landmark detector could later be used to initialize or correct the tracker.

---

## Algorithms Used

### 1. Shi–Tomasi Corner Detection

Detects strong, trackable feature points.

- We do **not** track polygon vertices directly.
- Points are detected in a thin ring around the polygon boundary, where ultrasound texture is strongest.

**Why:** The cavity interior is low-texture; edges are far more reliable.

---

### 2. Lucas–Kanade Optical Flow (Pyramidal)

Tracks feature points from one frame to the next.

- Uses image pyramids to handle larger motion.
- Runs efficiently in real time on CPU.

**Why:** Lightweight, proven, and well-suited for short-term motion tracking.

---

### 3. Forward–Backward Consistency Check

Each point is tracked:

- forward (frame _t_ → _t+1_)
- backward (_t+1_ → _t_)

Points that don’t return close to their original position are discarded.

**Why:** Ultrasound speckle often produces “confident but wrong” motion; this removes it.

---

### 4. RANSAC-Based Affine Motion Estimation

From the surviving point correspondences, we estimate a single affine transform:

- translation
- rotation
- scale

RANSAC automatically rejects outlier points.

**Why:** Local cardiac motion is well approximated by an affine model.

---

### 5. Least-Squares Refinement on Inliers

After RANSAC finds inliers:

- we re-fit the affine transform using only those inliers
- this improves precision and reduces jitter

**Why:** RANSAC is robust; least-squares is precise.

---

### 6. Template Matching (Recovery Mode)

When tracking confidence drops or motion is too large:

1. Search for the previous ROI in a **local expanded window**
2. If that fails, search the **entire frame at lower resolution**

Uses normalized cross-correlation.

**Why:** Optical flow assumes small motion; template matching recovers from sudden jumps.

---

### 7. Exponential Moving Average (EMA) Smoothing

We smooth:

- the affine motion parameters
- the final polygon vertices

**Why:** Reduces jitter while preserving real motion, which is important for demos and clinical usability.

---

## Main Pipeline Steps (End-to-End)

### 1. Initialization

- User draws a polygon on the first frame.
- We:
  - build a boundary ring mask
  - detect feature points in that ring
  - store a template patch for recovery

---

### 2. Frame Preprocessing

For each frame:

- convert to grayscale
- apply median filtering (reduce speckle)
- apply CLAHE (improve local contrast)

---

### 3. Feature Tracking

- Track feature points with pyramidal Lucas–Kanade optical flow
- Apply forward–backward filtering
- Discard unrealistic motion

---

### 4. Motion Estimation

- Estimate affine motion using RANSAC
- Refine using least-squares on inliers
- Clamp extreme translations for stability

If affine fails:

- fall back to robust translation
- optionally trigger recovery

---

### 5. Polygon Update

- Apply the estimated motion to the polygon
- Smooth polygon vertices with EMA

---

### 6. Recovery (When Needed)

Triggered by:

- low confidence
- too few valid points
- sudden large motion

Process:

1. Local template search
2. Global downsampled search if needed
3. Snap polygon back and reseed features

---

### 7. Output

Each frame produces:

- updated polygon points
- confidence score
- tracking mode (`affine`, `fallback`, `recovery`)
- events (lost, recovered, out-of-frame)

This makes the system observable, debuggable, and easy to integrate.

---

## Summary

This tracker combines well-established classical vision techniques into a robust, real-time system that works under difficult ultrasound conditions. It prioritizes interpretability, speed, and reliability, while leaving a clear path for future machine learning integration.
