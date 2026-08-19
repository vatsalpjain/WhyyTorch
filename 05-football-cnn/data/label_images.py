"""Manual bounding-box labeling for FIFA ball detection.

Positive images: drag a box around the ball.
Negative images: mark as no-ball (confidence target = 0).

Output: data/labels.json

Controls
--------
Left mouse drag : draw bounding box
s               : save current label
n / Right arrow : next image
p / Left arrow  : previous image
b               : mark image as no-ball
r               : reset current box
q               : quit (saves progress)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "images"
LABELS_PATH = Path(__file__).resolve().parent / "labels.json"


@dataclass
class Label:
    image: str
    has_ball: bool
    confidence: float
    # Normalized center format: easier for CNN regression than corner coords.
    cx: float | None = None
    cy: float | None = None
    w: float | None = None
    h: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Labeler:
    def __init__(self, image_paths: list[Path], labels: dict[str, Label]) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.index = 0

        self.drawing = False
        self.start_point: tuple[int, int] | None = None
        self.end_point: tuple[int, int] | None = None
        self.current_image: np.ndarray | None = None
        self.display_image: np.ndarray | None = None

        self.window_name = "FIFA Ball Labeler"

    def _load_existing_box(self) -> None:
        name = self.image_paths[self.index].name
        label = self.labels.get(name)
        if label is None or not label.has_ball or label.cx is None:
            self.start_point = None
            self.end_point = None
            return

        image = self.current_image
        assert image is not None
        h, w = image.shape[:2]
        half_w = (label.w or 0.0) * w / 2.0
        half_h = (label.h or 0.0) * h / 2.0
        cx = (label.cx or 0.0) * w
        cy = (label.cy or 0.0) * h
        self.start_point = (int(cx - half_w), int(cy - half_h))
        self.end_point = (int(cx + half_w), int(cy + half_h))

    def _refresh_display(self) -> None:
        assert self.current_image is not None
        self.display_image = self.current_image.copy()

        name = self.image_paths[self.index].name
        label = self.labels.get(name)
        status = "NO BALL" if label and not label.has_ball else "BALL"
        cv2.putText(
            self.display_image,
            f"{self.index + 1}/{len(self.image_paths)} {name} [{status}]",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        if self.start_point and self.end_point:
            cv2.rectangle(self.display_image, self.start_point, self.end_point, (0, 255, 0), 2)

        cv2.imshow(self.window_name, self.display_image)

    def _load_current_image(self) -> None:
        path = self.image_paths[self.index]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        self.current_image = image
        self._load_existing_box()
        self._refresh_display()

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
            self._refresh_display()
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            self._refresh_display()

    def _box_to_label(self) -> Label | None:
        if self.current_image is None or not self.start_point or not self.end_point:
            return None

        x1, y1 = self.start_point
        x2, y2 = self.end_point
        x_min, x_max = sorted((x1, x2))
        y_min, y_max = sorted((y1, y2))

        if x_max - x_min < 2 or y_max - y_min < 2:
            return None

        img_h, img_w = self.current_image.shape[:2]
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h
        cx = ((x_min + x_max) / 2.0) / img_w
        cy = ((y_min + y_max) / 2.0) / img_h

        name = self.image_paths[self.index].name
        return Label(
            image=name,
            has_ball=True,
            confidence=1.0,
            cx=round(cx, 6),
            cy=round(cy, 6),
            w=round(w, 6),
            h=round(h, 6),
        )

    def _save_current(self) -> None:
        name = self.image_paths[self.index].name
        existing = self.labels.get(name)
        if existing and not existing.has_ball:
            print(f"Saved no-ball label for {name}")
            return

        label = self._box_to_label()
        if label is None:
            print("Draw a box first, or press 'b' for no-ball images.")
            return

        self.labels[name] = label
        print(f"Saved ball label for {name}: cx={label.cx}, cy={label.cy}, w={label.w}, h={label.h}")

    def _mark_no_ball(self) -> None:
        name = self.image_paths[self.index].name
        self.labels[name] = Label(
            image=name,
            has_ball=False,
            confidence=0.0,
            cx=None,
            cy=None,
            w=None,
            h=None,
        )
        self.start_point = None
        self.end_point = None
        print(f"Marked {name} as no-ball")
        self._refresh_display()

    def _persist(self) -> None:
        ordered = [self.labels[p.name].to_dict() for p in self.image_paths if p.name in self.labels]
        LABELS_PATH.write_text(json.dumps(ordered, indent=2))
        print(f"Wrote {len(ordered)} labels to {LABELS_PATH}")

    def run(self) -> None:
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        self._load_current_image()

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                self._persist()
                break
            if key == ord("s"):
                self._save_current()
                self._persist()
            elif key in (ord("n"), 83):
                self.index = min(self.index + 1, len(self.image_paths) - 1)
                self._load_current_image()
            elif key in (ord("p"), 81):
                self.index = max(self.index - 1, 0)
                self._load_current_image()
            elif key == ord("b"):
                self._mark_no_ball()
                self._persist()
            elif key == ord("r"):
                self.start_point = None
                self.end_point = None
                self._refresh_display()

        cv2.destroyAllWindows()


def load_labels() -> dict[str, Label]:
    if not LABELS_PATH.exists():
        return {}

    raw = json.loads(LABELS_PATH.read_text())
    labels: dict[str, Label] = {}
    for item in raw:
        labels[item["image"]] = Label(**item)
    return labels


def main() -> None:
    image_paths = sorted(DATA_DIR.glob("*.png"), key=lambda p: int(p.stem.split("_")[1]))
    if not image_paths:
        raise SystemExit(f"No images found in {DATA_DIR}")

    labels = load_labels()
    labeler = Labeler(image_paths, labels)
    labeler.run()


if __name__ == "__main__":
    main()
