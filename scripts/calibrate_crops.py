#!/usr/bin/env python3
"""Interactive HUD crop calibration tool for Dota 2 screenshots.

Displays a reference screenshot and guides the user through marking
16 HUD regions (10 hero portraits + 6 item slots) via click-and-drag.
Saves coordinates to a JSON config file.

Usage:
    python scripts/calibrate_crops.py --image data/reference_screenshot.png --output configs/crop_config.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

REGION_ORDER = [
    "radiant_hero_1", "radiant_hero_2", "radiant_hero_3", "radiant_hero_4", "radiant_hero_5",
    "dire_hero_1", "dire_hero_2", "dire_hero_3", "dire_hero_4", "dire_hero_5",
    "item_slot_1", "item_slot_2", "item_slot_3", "item_slot_4", "item_slot_5", "item_slot_6",
]

COLORS = {
    "radiant": "#00FF00",
    "dire": "#FF4444",
    "item": "#00BFFF",
}


def get_color(name: str) -> str:
    if name.startswith("radiant"):
        return COLORS["radiant"]
    if name.startswith("dire"):
        return COLORS["dire"]
    return COLORS["item"]


class CropCalibrator:
    def __init__(self, image_path: str, output_path: str):
        self.image = Image.open(image_path)
        self.width, self.height = self.image.size
        self.output_path = output_path

        self.regions: dict[str, dict] = {}
        self.undo_stack: list[str] = []
        self.redo_stack: list[str] = []

        # Drawing state
        self.drag_start = None
        self.preview_rect = None

        # Matplotlib setup
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 9))
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
        self.ax.imshow(self.image)
        self.ax.set_axis_off()

        # Stored rectangle patches keyed by region name
        self.rect_patches: dict[str, tuple[patches.Rectangle, plt.Text]] = {}

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._update_title()

    @property
    def current_index(self) -> int:
        return len(self.regions)

    @property
    def current_name(self) -> str | None:
        if self.current_index < len(REGION_ORDER):
            return REGION_ORDER[self.current_index]
        return None

    def _update_title(self):
        if self.current_name:
            self.fig.suptitle(
                f"Draw: {self.current_name}  ({self.current_index + 1}/{len(REGION_ORDER)})  |  "
                f"[u]ndo  [r]edo  [s]ave & quit  [q]uit",
                fontsize=13, fontweight="bold",
            )
        else:
            self.fig.suptitle(
                f"All {len(REGION_ORDER)} regions defined!  |  [s]ave & quit  [u]ndo  [q]uit",
                fontsize=13, fontweight="bold", color="green",
            )
        self.fig.canvas.draw_idle()

    def _draw_rect(self, name: str, coords: dict):
        color = get_color(name)
        rect = patches.Rectangle(
            (coords["x"], coords["y"]), coords["w"], coords["h"],
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.2,
        )
        self.ax.add_patch(rect)
        text = self.ax.text(
            coords["x"] + 4, coords["y"] + 18, name,
            color=color, fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
        )
        self.rect_patches[name] = (rect, text)
        self.fig.canvas.draw_idle()

    def _remove_rect(self, name: str):
        if name in self.rect_patches:
            rect, text = self.rect_patches.pop(name)
            rect.remove()
            text.remove()
            self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax or self.current_name is None:
            return
        if event.button != 1:
            return
        self.drag_start = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if self.drag_start is None or event.inaxes != self.ax:
            return
        # Update preview rectangle
        if self.preview_rect:
            self.preview_rect.remove()
        x0, y0 = self.drag_start
        x1, y1 = event.xdata, event.ydata
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        color = get_color(self.current_name)
        self.preview_rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor=color, facecolor="none", linestyle="--",
        )
        self.ax.add_patch(self.preview_rect)
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self.drag_start is None or self.current_name is None:
            return
        if event.inaxes != self.ax:
            self.drag_start = None
            if self.preview_rect:
                self.preview_rect.remove()
                self.preview_rect = None
                self.fig.canvas.draw_idle()
            return

        # Remove preview
        if self.preview_rect:
            self.preview_rect.remove()
            self.preview_rect = None

        x0, y0 = self.drag_start
        x1, y1 = event.xdata, event.ydata
        self.drag_start = None

        # Ignore tiny drags (accidental clicks)
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            self.fig.canvas.draw_idle()
            return

        x, y = int(min(x0, x1)), int(min(y0, y1))
        w, h = int(abs(x1 - x0)), int(abs(y1 - y0))

        name = self.current_name
        self.regions[name] = {"x": x, "y": y, "w": w, "h": h}
        self.undo_stack.append(name)
        self.redo_stack.clear()

        self._draw_rect(name, self.regions[name])
        self._update_title()

        # Auto-save when all regions are defined
        if self.current_name is None:
            self._save()

    def _on_key(self, event):
        if event.key == "u":
            self._undo()
        elif event.key == "r":
            self._redo()
        elif event.key == "s":
            self._save()
            plt.close(self.fig)
        elif event.key == "q":
            plt.close(self.fig)

    def _undo(self):
        if not self.undo_stack:
            return
        name = self.undo_stack.pop()
        self.redo_stack.append((name, self.regions.pop(name)))
        self._remove_rect(name)
        self._update_title()

    def _redo(self):
        if not self.redo_stack:
            return
        name, coords = self.redo_stack.pop()
        self.regions[name] = coords
        self.undo_stack.append(name)
        self._draw_rect(name, coords)
        self._update_title()

    def _save(self):
        config = {
            "reference_resolution": [self.width, self.height],
            "regions": self.regions,
        }
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved {len(self.regions)} regions to {self.output_path}")

    def run(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Calibrate HUD crop regions on a Dota 2 screenshot")
    parser.add_argument("--image", required=True, help="Path to reference screenshot")
    parser.add_argument("--output", default="configs/crop_config.json", help="Output JSON path")
    args = parser.parse_args()

    calibrator = CropCalibrator(args.image, args.output)
    calibrator.run()


if __name__ == "__main__":
    main()
