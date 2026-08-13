"""
Template-based mark detection: learns the printed option letters from the
sheet itself and detects only ink that is EXTRA relative to that printed
template - i.e. the student's added mark, not the printed "a)"/"b)"/"c)"/"d)"
glyph that's on every sheet regardless of what was marked.

This replaces the earlier global ink-ratio/fixed-threshold detector: that
approach compared every option against one fixed number, which broke down
because the printed glyphs themselves ("b" and "d" in particular) contain
more ink than "a" and "c" purely from font shape - not from any mark. A
single threshold could never be both low enough to catch a faint mark and
high enough to ignore that baseline glyph noise. Subtracting the learned
template removes that baseline entirely before any decision is made.
"""

import cv2
import numpy as np

MARK_PIXEL_THRESHOLD = 15
INK_THRESHOLD = MARK_PIXEL_THRESHOLD  # backward-compatible alias


def ink_ratio(binary_img, box, pad_left=16, pad=4):
    x, y, w, h = box
    x = int(round(x - pad_left))
    y = int(round(y - pad))
    w = int(round(w + pad_left + pad))
    h = int(round(h + 2 * pad))
    roi = binary_img[max(y, 0): y + h, max(x, 0): x + w]
    if roi.size == 0:
        return 0.0
    return cv2.countNonZero(roi) / roi.size


def build_letter_templates(binary_img, layout):
    """Build one printed-letter template for a, b, c, d from the least-ink occurrence."""
    candidates = {letter: [] for letter in "abcd"}

    for qnum, question_layout in layout["questions"].items():
        for letter, boxes in question_layout.items():
            x, y, w, h = map(int, boxes["letter_box"])
            crop = binary_img[y:y + h, x:x + w]
            if crop.shape != (h, w):
                continue
            ratio = cv2.countNonZero(crop) / float(crop.size)
            candidates.setdefault(letter, []).append((ratio, str(qnum), crop.copy()))

    templates = {}
    sources = {}

    for letter in "abcd":
        items = candidates.get(letter, [])
        if not items:
            raise ValueError(f"No letter-box samples found for option {letter!r}.")

        # The unmarked printed glyph has the least ink. Marks add pixels, so
        # selecting the minimum-ink occurrence gives a clean template on the
        # current generated sheets without requiring a separate blank upload.
        ratio, qnum, crop = min(items, key=lambda item: item[0])
        templates[letter] = (crop > 0).astype(np.uint8)
        sources[letter] = {"question": qnum, "ink_ratio": float(ratio)}

    return templates, sources


def prepare_mark_detector(binary_img, layout):
    templates, sources = build_letter_templates(binary_img, layout)

    # Remove the printed a/b/c/d glyph from the binary sheet. Everything left
    # around the option positions is candidate student-added ink.
    printed = np.zeros_like(binary_img, dtype=np.uint8)

    option_centers = []
    option_meta = []

    for qnum, question_layout in layout["questions"].items():
        for letter, boxes in question_layout.items():
            x, y, w, h = map(int, boxes["letter_box"])
            template = templates[letter]
            printed[y:y + h, x:x + w] = np.maximum(
                printed[y:y + h, x:x + w], template * 255
            )
            option_centers.append((x + w / 2.0, y + h / 2.0))
            option_meta.append((str(qnum), letter, boxes))

    candidate_ink = (binary_img > 0) & (printed == 0)
    centers = np.asarray(option_centers, dtype=np.float32)

    return {
        "templates": templates,
        "template_sources": sources,
        "candidate_ink": candidate_ink,
        "centers": centers,
        "option_meta": option_meta,
    }


def detect_marks_for_question(binary_img, question_layout, detector, ink_thresh=None):
    """Detect marks by counting student-added pixels assigned to each option."""
    threshold = MARK_PIXEL_THRESHOLD if ink_thresh is None else int(ink_thresh)
    candidate_ink = detector["candidate_ink"]
    centers = detector["centers"]
    option_meta = detector["option_meta"]

    # Assign every candidate pixel to its nearest option center. A 15-pixel
    # radius prevents nearby printed text or another option's mark from being
    # counted for this option.
    yy, xx = np.where(candidate_ink)
    scores = {letter: 0 for letter in question_layout}

    if len(xx):
        points = np.column_stack((xx, yy)).astype(np.float32)
        distances = np.sqrt(((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        nearest = distances.argmin(axis=1)
        nearest_distance = distances[np.arange(len(points)), nearest]

        target_indices = set()
        target_boxes = {
            letter: tuple(map(int, boxes["letter_box"]))
            for letter, boxes in question_layout.items()
        }

        for option_index, (_qnum, letter, boxes) in enumerate(option_meta):
            if letter in target_boxes and tuple(map(int, boxes["letter_box"])) == target_boxes[letter]:
                target_indices.add(option_index)

        for idx in np.where(nearest_distance <= 15.0)[0]:
            option_index = int(nearest[idx])
            if option_index in target_indices:
                _qnum, letter, _ = option_meta[option_index]
                scores[letter] = scores.get(letter, 0) + 1

    results = {}
    for letter, boxes in question_layout.items():
        # Diagnostic ratios only - NOT used for the mark decision anymore.
        x, y, w, h = map(int, boxes["letter_box"])
        letter_roi = binary_img[y:y + h, x:x + w]
        letter_ratio = (cv2.countNonZero(letter_roi) / float(letter_roi.size)) if letter_roi.size else 0.0
        line_ratio = ink_ratio(binary_img, boxes["line_box"])
        added_pixels = int(scores.get(letter, 0))

        results[letter] = {
            "letter_ratio": letter_ratio,
            "line_ratio": line_ratio,
            "added_pixels": added_pixels,
            "marked": added_pixels >= threshold,
        }

    return results
