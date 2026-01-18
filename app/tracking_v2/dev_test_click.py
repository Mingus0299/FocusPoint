import cv2
import numpy as np
from datetime import datetime

from tracker_lk import LKTracker

TARGET_W, TARGET_H = 1100, 850  # output canvas/window size


def letterbox(image, target_w, target_h):
    """Return (canvas, scale, xoff, yoff) where image is aspect-fit into canvas."""
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    xoff = (target_w - new_w) // 2
    yoff = (target_h - new_h) // 2
    canvas[yoff:yoff + new_h, xoff:xoff + new_w] = resized
    return canvas, scale, xoff, yoff


def to_frame_coords(x, y, scale, xoff, yoff):
    fx = (x - xoff) / scale
    fy = (y - yoff) / scale
    return fx, fy


def to_canvas_coords(x, y, scale, xoff, yoff):
    cx = x * scale + xoff
    cy = y * scale + yoff
    return cx, cy


def draw_polygon(img, pts, is_closed=False):
    """pts are in DISPLAY/CANVAS coords."""
    if len(pts) == 0:
        return

    for (x, y) in pts:
        cv2.circle(img, (int(x), int(y)), 4, (255, 255, 255), -1)

    if len(pts) >= 2:
        p = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [p], isClosed=is_closed, color=(0, 255, 0), thickness=2)


def make_writer(out_path, fps, w, h):
    # Use mp4v for broad compatibility
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(out_path, fourcc, fps, (w, h))

# TODO: update path
def main():
    # video_path = r"C:\Users\jocel\OneDrive\Desktop\Holoray_Dataset\Echo\echo4.mp4"
    # video_path = r"C:\Users\jocel\OneDrive\Desktop\Holoray_Dataset\Intrapartum\Intrapartum-occlusions.mp4"
    video_path = r"C:\Users\jocel\OneDrive\Desktop\Holoray_Dataset\Lapchole\Lapchole1.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    # Try to read video fps; fallback if missing
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 120:
        fps = 30.0

    if fps is None or fps <= 1e-6:
        fps = 30.0

    ok, frame0 = cap.read()
    if not ok:
        print(f"Could not read first frame: {video_path}")
        return

    win = "Draw polygon: click points | Enter=done | Backspace=undo | Esc=clear"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, TARGET_W, TARGET_H)

    polygon = []  # FRAME coords
    finalized = False

    scale, xoff, yoff = 1.0, 0, 0

    def on_mouse(event, x, y, flags, param):
        nonlocal polygon, finalized, scale, xoff, yoff
        if finalized:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            fx, fy = to_frame_coords(x, y, scale, xoff, yoff)
            h0, w0 = frame0.shape[:2]
            if 0 <= fx < w0 and 0 <= fy < h0:
                polygon.append((float(fx), float(fy)))

    cv2.setMouseCallback(win, on_mouse)

    # --- Phase 1: polygon drawing ---
    while True:
        vis, scale, xoff, yoff = letterbox(frame0, TARGET_W, TARGET_H)
        disp_poly = [to_canvas_coords(x, y, scale, xoff, yoff) for x, y in polygon]
        draw_polygon(vis, disp_poly, is_closed=(len(disp_poly) >= 3))

        cv2.putText(vis, f"Points: {len(polygon)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(vis, "Click vertices. Enter=start. Backspace=undo. Esc=clear. q=quit",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(win, vis)
        key = cv2.waitKey(16) & 0xFF

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return

        if key in (13, 10):  # Enter
            if len(polygon) >= 3:
                finalized = True
                break

        if key == 8:  # Backspace
            if polygon:
                polygon.pop()

        if key == 27:  # Esc
            polygon = []
    
    print("Polygon (frame coords):", polygon)

    # --- Phase 2: tracking ---
    tracker = LKTracker()
    tracker.init(frame0, polygon)

    # Create output writer (saves what you SEE: letterboxed canvas with overlay)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"tracked_{timestamp}.mp4"
    writer = make_writer(out_path, fps, TARGET_W, TARGET_H)
    if not writer.isOpened():
        print("Warning: could not open VideoWriter. Video will not be saved.")
        writer = None
    else:
        print(f"Saving tracked video to: {out_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out = tracker.update(frame)
        pts = out.get("points", [])          # FRAME coords
        conf = float(out.get("confidence", 0.0))
        mode = out.get("mode", "flow")
        events = out.get("events", [])

        vis, scale, xoff, yoff = letterbox(frame, TARGET_W, TARGET_H)
        disp_pts = [to_canvas_coords(x, y, scale, xoff, yoff) for x, y in pts]
        draw_polygon(vis, disp_pts, is_closed=True)

        cv2.putText(vis, f"conf={conf:.2f} mode={mode}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if events:
            txt = " | ".join(events[:3])
            cv2.putText(vis, txt, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show + save
        cv2.imshow("LK Tracker Test (press q to quit)", vis)
        if writer is not None:
            writer.write(vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
