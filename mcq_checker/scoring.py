"""
Compares decoded student answers against the answer key and produces a
per-question result plus a final score summary.
"""


def compare_to_key(decoded: dict, answer_key: dict) -> dict:
    results = {}
    for qnum in sorted(answer_key, key=int):
        student_data = decoded.get(qnum)
        student_answer = student_data["answer"] if student_data else None
        correct_answer = answer_key[qnum]
        result = "CORRECT" if student_answer == correct_answer else "WRONG"

        results[qnum] = {
            "student_answer": student_answer,
            "correct_answer": correct_answer,
            "result": result,
        }
    return results


def summarize(results: dict) -> dict:
    total_questions = len(results)
    correct = sum(1 for r in results.values() if r["result"] == "CORRECT")
    wrong = total_questions - correct
    percentage = (correct / total_questions) * 100 if total_questions > 0 else 0

    return {
        "total_questions": total_questions,
        "correct": correct,
        "wrong": wrong,
        "percentage": percentage,
    }
