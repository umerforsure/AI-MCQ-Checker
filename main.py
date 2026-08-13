#!/usr/bin/env python3
"""
AI MCQ Checker - main entry point.

Runs the full checking pipeline on one student sheet:
  1. Load the exam layout (by exam_id, from local storage)
  2. Load the answer key (JSON)
  3. Read the student's filled sheet image
  4. Orientation correction (fix rotation/flip via ArUco corner markers)
  5. Preprocessing (grayscale, denoise, binarize)
  6. Mark detection (template-based - subtracts the printed glyph, counts
     only the student's added ink)
  7. Answer decoding (single letter / blank / MULTI)
  8. Score against the answer key

Usage:
    python main.py --exam-id test_fixture_general_science \\
                    --key sample_data/test_sheet_answer_key.json \\
                    --student sample_data/test_sheet_filled.png

    # first time only, to register a layout the generator side produced:
    python main.py --register-layout sample_data/test_sheet_layout.json
"""

import argparse
import json
import os
import sys

import cv2

from mcq_checker.layout_store import get_stored_layout, register_layout, validate_layout
from mcq_checker.orientation import correct_orientation, OrientationError
from mcq_checker.preprocessing import preprocess
from mcq_checker.mark_detection import prepare_mark_detector, MARK_PIXEL_THRESHOLD
from mcq_checker.decode import decode_answers
from mcq_checker.scoring import compare_to_key, summarize


def load_answer_key(key_path: str) -> dict:
    ext = os.path.splitext(key_path)[1].lower()

    if ext != ".json":
        raise SystemExit(
            f"Answer key must be a JSON file for now (got {ext}). "
            "Image-based answer key derivation isn't implemented yet."
        )

    with open(key_path) as f:
        key_data = json.load(f)

    return key_data["key"] if "key" in key_data else key_data


def save_debug_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)
    print(f"  wrote {path}")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def resolve_student_images(path: str) -> list:
    """
    --student can point to either:
      - a single image file  -> checks just that one student
      - a folder of images   -> checks every sheet inside, one run each
    Returns a sorted list of image file paths either way.
    """
    if os.path.isdir(path):
        files = [
            os.path.join(path, f)
            for f in sorted(os.listdir(path))
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
        if not files:
            raise SystemExit(f"No image files found in folder: {path}")
        return files
    elif os.path.isfile(path):
        return [path]
    else:
        raise SystemExit(f"Student sheet path not found: {path}")


def process_one_student(student_path, layout, answer_key, args):
    """Runs the full pipeline (orientation -> preprocess -> detect -> decode -> score) on one sheet."""
    student_name = os.path.splitext(os.path.basename(student_path))[0]

    student_img = cv2.imread(student_path)
    if student_img is None or student_img.size == 0:
        raise SystemExit(f"Could not read student sheet as a valid image: {student_path}")
    print(f"\n--- {student_name} ---")
    print(f"Student image loaded: {student_path} ({student_img.shape[1]}x{student_img.shape[0]})")

    try:
        corrected_img, orient_info = correct_orientation(student_img)
    except OrientationError as e:
        raise SystemExit(f"Orientation error ({student_name}): {e}")
    print(f"Orientation corrected: {orient_info}")

    if args.debug_images:
        save_debug_image(corrected_img, os.path.join(args.outdir, f"{student_name}_01_corrected.png"))

    gray, binary = preprocess(
        corrected_img,
        denoise_method=args.denoise_method,
        binarize_method=args.binarize_method,
    )
    print("Preprocessing complete "
          f"(denoise={args.denoise_method}, binarize={args.binarize_method})")

    if args.debug_images:
        save_debug_image(binary, os.path.join(args.outdir, f"{student_name}_02_binary.png"))

    detector = prepare_mark_detector(binary, layout)

    decoded = decode_answers(binary, layout, detector, ink_thresh=MARK_PIXEL_THRESHOLD)
    print("Decoded answers:")
    for qnum in sorted(decoded, key=int):
        print(f"  {qnum} -> {decoded[qnum]['answer']}")

    results = compare_to_key(decoded, answer_key)
    print("Results:")
    for qnum, r in results.items():
        print(f"  {qnum} -> Student: {r['student_answer']}, "
              f"Correct: {r['correct_answer']}, {r['result']}")

    summary = summarize(results)
    print(f"Score: {summary['correct']}/{summary['total_questions']} "
          f"({summary['percentage']:.2f}%)")

    return student_name, results, summary

def run(args):
    # ---- 1. Register layout (optional one-off command) ----
    if args.register_layout:
        exam_id = register_layout(args.register_layout)
        print(f"Registered layout for exam_id={exam_id!r}")
        if not args.exam_id:
            return
        # fall through if the user also asked to run a check in the same call

    # ---- 2. Load + validate layout ----
    if not args.exam_id:
        raise SystemExit("Missing --exam-id (or use --register-layout by itself first).")

    try:
        layout = get_stored_layout(args.exam_id)
        validate_layout(layout)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"Layout error: {e}")

    print(f"Layout loaded: {layout['exam_id']} ({len(layout['questions'])} questions)")

    # ---- 3. Load answer key ----
    if not args.key:
        raise SystemExit("Missing --key (path to answer key JSON).")
    answer_key = load_answer_key(args.key)
    print(f"Answer key loaded: {answer_key}")

   # ---- 4. Resolve student input: single file or a whole folder ----
    if not args.student:
        raise SystemExit("Missing --student (path to a sheet image, or a folder of them).")
    student_images = resolve_student_images(args.student)
    print(f"Found {len(student_images)} student sheet(s) to check.")

    # ---- 5-8. Run the pipeline once per student ----
    all_results = {}
    for student_path in student_images:
        student_name, results, summary = process_one_student(student_path, layout, answer_key, args)
        all_results[student_name] = {"results": results, "summary": summary}

    # ---- Class-level summary when checking more than one sheet ----
    if len(student_images) > 1:
        print("\n" + "=" * 40)
        print("CLASS SUMMARY")
        print("=" * 40)
        for name, r in all_results.items():
            s = r["summary"]
            print(f"  {name}: {s['correct']}/{s['total_questions']} ({s['percentage']:.2f}%)")

    # ---- Output JSON: combined for a folder run, single-file behavior otherwise ----
    if args.output_json:
        if len(student_images) > 1:
            os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nWrote combined results for {len(student_images)} students to {args.output_json}")
        else:
            single_name = next(iter(all_results))
            os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(all_results[single_name], f, indent=2)
            print(f"\nWrote results to {args.output_json}")


def parse_args():
    p = argparse.ArgumentParser(description="AI MCQ Checker")
    p.add_argument("--exam-id", help="Exam ID to look up the stored layout for.")
    p.add_argument("--register-layout", help="Path to a layout JSON to register (DEV-only, "
                                              "stand-in for the generator system).")
    p.add_argument("--key", help="Path to the answer key JSON.")
    p.add_argument("--student", help="Path to the student's filled sheet image, or a folder of them.")
    p.add_argument("--denoise-method", default="nlmeans",
                    choices=["auto", "median", "nlmeans"])
    p.add_argument("--binarize-method", default="otsu", choices=["otsu", "adaptive"])
    p.add_argument("--debug-images", action="store_true",
                    help="Save intermediate images (orientation-corrected, binarized) to --outdir.")
    p.add_argument("--outdir", default="output", help="Directory for debug images.")
    p.add_argument("--output-json", help="Optional path to write results as JSON.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())