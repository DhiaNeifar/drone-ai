#!/usr/bin/env python3
"""Browse GEM videos with Label Studio video rectangles overlaid."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2


WINDOW_NAME = "GEM video annotations"
LEFT_KEYS = {2, 81, 63234, 65361, 2424832}
RIGHT_KEYS = {3, 83, 63235, 65363, 2555904}
UP_KEYS = {0, 82, 63232, 65362, 2490368}
DOWN_KEYS = {1, 84, 63233, 65364, 2621440}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse GEM source videos with Label Studio boxes overlaid."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/gem_new_videos"),
        help="Folder containing export_label_studio.json and videos/.",
    )
    parser.add_argument("--video", help="Only view videos whose filename contains this text.")
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument(
        "--start-frame",
        type=int,
        help="Initial frame (default: 0).",
    )
    start_group.add_argument(
        "--first-annotation",
        action="store_true",
        help="Start each video at its first active bounding-box annotation.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Initial playback speed multiplier (default: 1).",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="Open paused instead of playing automatically.",
    )
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--max-height", type=int, default=900)
    return parser.parse_args()


def find_video(videos_dir: Path, video_url: str) -> Path | None:
    exported_name = Path(video_url).name
    candidates = [exported_name]
    if "-" in exported_name:
        candidates.append(exported_name.split("-", 1)[1])
    files = {path.name: path for path in videos_dir.rglob("*") if path.is_file()}
    return next((files[name] for name in candidates if name in files), None)


def drone_tracks(task: dict) -> list[list[dict]]:
    tracks = []
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            value = result.get("value", {})
            if result.get("type") != "videorectangle" or "Drone" not in value.get("labels", []):
                continue
            sequence = sorted(value.get("sequence", []), key=lambda point: float(point["time"]))
            if sequence:
                tracks.append(sequence)
    return tracks


def first_active_time(tracks: list[list[dict]]) -> float:
    times = [
        float(point["time"])
        for track in tracks
        for point in track
        if point.get("enabled", False)
    ]
    return min(times, default=0.0)


def interpolate(timestamp: float, track: list[dict]) -> tuple[float, ...] | None:
    before = None
    after = None
    for point in track:
        time = float(point["time"])
        if time <= timestamp:
            before = point
        if time >= timestamp and after is None:
            after = point
    if before is None or not before.get("enabled", False):
        return None
    if after is None or float(after["time"]) == float(before["time"]):
        point = before
        return tuple(float(point[key]) for key in ("x", "y", "width", "height"))
    ratio = (timestamp - float(before["time"])) / (
        float(after["time"]) - float(before["time"])
    )
    return tuple(
        float(before[key]) + (float(after[key]) - float(before[key])) * ratio
        for key in ("x", "y", "width", "height")
    )


def fit(image, max_width: int, max_height: int):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def draw(frame, tracks: list[list[dict]], timestamp: float, status: str, speed: float):
    height, width = frame.shape[:2]
    count = 0
    for track in tracks:
        box = interpolate(timestamp, track)
        if box is None:
            continue
        x, y, box_width, box_height = box
        x1, y1 = round(x * width / 100), round(y * height / 100)
        x2, y2 = round((x + box_width) * width / 100), round((y + box_height) * height / 100)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "drone", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        count += 1
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 66), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, f"{status} | boxes={count} | speed={speed:g}x", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Left/Right: frame | Up/Down: video | +/-: speed | Space: pause | Q: quit", (12, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    tasks = json.loads((root / "export_label_studio.json").read_text(encoding="utf-8"))
    entries = []
    for task in tasks:
        video_url = task.get("data", {}).get("video_url", "")
        video_path = find_video(root / "videos", video_url)
        if video_path and (not args.video or args.video.casefold() in video_path.name.casefold()):
            entries.append((video_path, drone_tracks(task)))
    if not entries:
        raise RuntimeError(f"No matching annotated videos under {root}")
    if args.stride < 1 or args.speed <= 0 or args.max_width < 1 or args.max_height < 1:
        raise ValueError("--stride, --speed, and display dimensions must be positive")

    video_index = 0
    frame_positions = []
    captures = []
    fps_values = []
    frame_counts = []
    for video_path, tracks in entries:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
        captures.append(capture)
        fps_values.append(fps)
        frame_counts.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if not args.first_annotation:
            frame_positions.append(max(0, args.start_frame or 0))
            continue
        # Label Studio timestamps mark the end of a displayed video frame.
        frame_positions.append(max(0, round(first_active_time(tracks) * fps) - 1))
    playing = not args.paused
    playback_speed = args.speed
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            video_path, tracks = entries[video_index]
            capture = captures[video_index]
            fps = fps_values[video_index]
            frame_count = frame_counts[video_index]
            frame_index = min(frame_positions[video_index], max(0, frame_count - 1))
            if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame_index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
            timestamp = (frame_index + 1) / fps
            status = f"video {video_index + 1}/{len(entries)} {video_path.name} | frame {frame_index}/{frame_count - 1}"
            cv2.imshow(WINDOW_NAME, fit(draw(frame, tracks, timestamp, status, playback_speed), args.max_width, args.max_height))
            playback_step = max(1, math.floor(playback_speed)) * args.stride
            delay = max(1, round(1000 * playback_step / (fps * playback_speed)))
            key = cv2.waitKeyEx(delay if playing else 0)
            if key in (27, ord("q"), ord("Q")):
                break
            if key == 32:
                playing = not playing
            elif key in (ord("+"), ord("="), ord("]")):
                playback_speed = min(32.0, playback_speed * 2)
            elif key in (ord("-"), ord("_"), ord("[")):
                playback_speed = max(0.25, playback_speed / 2)
            elif key in LEFT_KEYS or key in (ord("a"), ord("A")):
                frame_positions[video_index] = max(0, frame_index - args.stride)
                playing = False
            elif key in RIGHT_KEYS or key in (ord("d"), ord("D")):
                frame_positions[video_index] = min(frame_count - 1, frame_index + args.stride)
                playing = False
            elif key in UP_KEYS or key in (ord("w"), ord("W")):
                video_index = max(0, video_index - 1)
            elif key in DOWN_KEYS or key in (ord("s"), ord("S")):
                video_index = min(len(entries) - 1, video_index + 1)
            elif playing:
                frame_positions[video_index] = frame_index + playback_step
                if frame_positions[video_index] >= frame_count:
                    frame_positions[video_index] = frame_count - 1
                    if video_index < len(entries) - 1:
                        video_index += 1
                    else:
                        playing = False
    finally:
        for capture in captures:
            capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
