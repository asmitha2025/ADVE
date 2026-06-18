"""
Generates a synthetic test video for ADVE validation.
Contains 3 objects moving in frame, plus 1 new object entering mid-video.
This validates both the delta path and Branch 2 (new object detection).
"""
import cv2
import numpy as np
import os


def generate(output_path: str = "test_video.mp4", duration_sec: int = 15, fps: int = 30):
    W, H = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    total_frames = duration_sec * fps
    branch2_frame = total_frames // 2  # New object enters at midpoint

    objects = [
        {
            "name":  "person",
            "color": (180, 80, 80),
            "pos":   [80.0, 100.0],
            "size":  (70, 130),
            "vel":   [1.8, 0.7],
        },
        {
            "name":  "laptop",
            "color": (80, 180, 80),
            "pos":   [300.0, 180.0],
            "size":  (110, 75),
            "vel":   [-1.2, 1.5],
        },
        {
            "name":  "chair",
            "color": (80, 80, 180),
            "pos":   [480.0, 280.0],
            "size":  (90, 110),
            "vel":   [0.9, -1.8],
        },
    ]

    for fi in range(total_frames):
        frame = np.full((H, W, 3), 30, dtype=np.uint8)  # dark background

        # Branch 2: new object enters at midpoint
        if fi == branch2_frame:
            objects.append({
                "name":  "bottle",
                "color": (200, 200, 60),
                "pos":   [10.0, 10.0],
                "size":  (35, 90),
                "vel":   [2.5, 1.2],
            })
            print(f"  [frame {fi:>4}] Branch 2 triggered — 'bottle' enters scene")

        for obj in objects:
            obj["pos"][0] += obj["vel"][0]
            obj["pos"][1] += obj["vel"][1]

            w, h = obj["size"]
            x, y = int(obj["pos"][0]), int(obj["pos"][1])

            # Bounce
            if x < 0 or x + w > W:
                obj["vel"][0] *= -1
                obj["pos"][0] = max(0.0, min(float(W - w), obj["pos"][0]))
            if y < 0 or y + h > H:
                obj["vel"][1] *= -1
                obj["pos"][1] = max(0.0, min(float(H - h), obj["pos"][1]))

            x, y = int(obj["pos"][0]), int(obj["pos"][1])

            # Draw filled rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), obj["color"], -1)
            # Darker border
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          tuple(max(0, c - 50) for c in obj["color"]), 2)
            # Label
            cv2.putText(
                frame, obj["name"],
                (x + 4, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA
            )

        # Frame counter overlay
        cv2.putText(
            frame, f"frame {fi:04d}", (8, H - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1
        )

        out.write(frame)

    out.release()
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\nGenerated: {output_path}  ({total_frames} frames, {size_mb:.1f} MB)")
    return output_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output",   default="test_video.mp4")
    p.add_argument("--duration", type=int, default=15)
    p.add_argument("--fps",      type=int, default=30)
    args = p.parse_args()

    generate(args.output, args.duration, args.fps)
