"""
Decodes each question to one answer: single letter, None (blank), or
"MULTI" (counts wrong - multiple selections are never valid on this format).
"""

from .mark_detection import detect_marks_for_question, MARK_PIXEL_THRESHOLD


def decode_answers(binary_img, layout, detector, ink_thresh=None):
    if ink_thresh is None:
        ink_thresh = MARK_PIXEL_THRESHOLD

    decoded = {}
    for qnum, question_layout in layout["questions"].items():
        detail = detect_marks_for_question(binary_img, question_layout, detector, ink_thresh)

        marked_letters = [letter for letter, result in detail.items() if result["marked"]]

        if len(marked_letters) == 1:
            answer = marked_letters[0]
        elif len(marked_letters) == 0:
            answer = None
        else:
            answer = "MULTI"

        decoded[qnum] = {"answer": answer, "detail": detail}

    return decoded
