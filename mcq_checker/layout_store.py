"""
Layout storage and validation.

get_stored_layout() loads a saved layout JSON by exam_id from local storage
(the `layouts/` folder). This is a stand-in for the real generator system's
database - once the two projects are integrated, this function's internals
get swapped for a real DB/API call, but the rest of the pipeline doesn't
need to change since it only ever calls get_stored_layout(exam_id).

register_layout() copies an externally-supplied layout JSON into local
storage, keyed by the exam_id found inside it. This is the DEV-ONLY stand-in
for "the generator system already has this" - once integrated, nothing
in this checking project registers layouts anymore, the generator side
writes directly to wherever the real store ends up being.
"""

import os
import json
import shutil

LAYOUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "layouts")


def get_stored_layout(exam_id: str) -> dict:
    exam_id = exam_id.strip()

    if not exam_id:
        raise ValueError("Exam ID cannot be empty.")

    if not exam_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Exam ID contains invalid characters.")

    layout_path = os.path.join(LAYOUT_DIR, f"{exam_id}.json")

    if not os.path.isfile(layout_path):
        raise FileNotFoundError(f"No layout found for exam ID: {exam_id}")

    try:
        with open(layout_path, "r", encoding="utf-8") as f:
            layout = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Layout file is not valid JSON: {layout_path}")

    if not isinstance(layout, dict):
        raise ValueError("Layout must be a JSON object.")

    if layout.get("exam_id") != exam_id:
        raise ValueError(f"Layout exam_id does not match requested exam ID: {exam_id}")

    if "questions" not in layout:
        raise ValueError("Layout is missing 'questions'.")

    if not isinstance(layout["questions"], dict) or not layout["questions"]:
        raise ValueError("Layout contains no questions.")

    return layout


def register_layout(layout_file_path: str) -> str:
    """
    DEV-ONLY: copies an external layout JSON into local storage under its
    own exam_id, so get_stored_layout() can find it afterward.
    Returns the exam_id it was registered under.
    """
    os.makedirs(LAYOUT_DIR, exist_ok=True)

    with open(layout_file_path) as f:
        uploaded_layout = json.load(f)

    exam_id_from_file = uploaded_layout.get("exam_id")
    if not exam_id_from_file:
        raise ValueError("Layout JSON has no exam_id.")

    dest_path = os.path.join(LAYOUT_DIR, f"{exam_id_from_file}.json")
    shutil.copy(layout_file_path, dest_path)
    return exam_id_from_file


def validate_layout(layout: dict) -> bool:
    if not isinstance(layout, dict):
        raise ValueError("Layout must be a JSON object.")

    if not layout.get("exam_id"):
        raise ValueError("Layout is missing exam_id.")

    questions = layout.get("questions")

    if not isinstance(questions, dict) or not questions:
        raise ValueError("Layout contains no questions.")

    for qnum, question in questions.items():
        if not isinstance(question, dict) or not question:
            raise ValueError(f"Question {qnum} has no options.")

        for option, boxes in question.items():
            if not isinstance(boxes, dict):
                raise ValueError(f"Question {qnum}, option {option}: invalid box data.")

            for box_name in ("letter_box", "line_box"):
                box = boxes.get(box_name)

                if not isinstance(box, list) or len(box) != 4:
                    raise ValueError(f"Question {qnum}, option {option}: invalid {box_name}.")

                if not all(isinstance(v, (int, float)) for v in box):
                    raise ValueError(
                        f"Question {qnum}, option {option}: {box_name} contains invalid coordinates."
                    )

                x, y, w, h = box
                if w <= 0 or h <= 0:
                    raise ValueError(
                        f"Question {qnum}, option {option}: {box_name} has invalid dimensions."
                    )

    return True
