# AI MCQ Checker

Checking/grading side of the AI MCQ project. Takes a student's filled
answer sheet (image) plus an exam layout and answer key, and produces a
score - no OCR, no LLM, only ArUco-marker alignment and pixel-level mark
detection.

## Project structure

```
mcq_checker_project/
├── main.py                    # entry point - run this
├── requirements.txt
├── mcq_checker/
│   ├── layout_store.py        # load/validate/register exam layouts
│   ├── orientation.py         # ArUco-based rotation/flip correction
│   ├── preprocessing.py       # grayscale, denoise, binarize
│   ├── mark_detection.py      # template-based mark detection
│   ├── decode.py              # per-question answer decoding
│   └── scoring.py             # compare to answer key, summarize
├── layouts/                   # registered exam layouts (by exam_id) - DEV-ONLY
│                               # local stand-in for the generator system's
│                               # database, until the two projects integrate
├── sample_data/                # test layout / key / student sheet
└── output/                    # debug images + JSON results land here
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**First time only** - register a layout (stand-in for what the generator
system will provide automatically once integrated):

```bash
python main.py --register-layout sample_data/test_sheet_layout.json
```

**Run a check:**

```bash
python main.py \
  --exam-id test_fixture_general_science \
  --key sample_data/test_sheet_answer_key.json \
  --student sample_data/test_sheet_filled.png \
  --debug-images \
  --output-json output/results.json
```

## CLI options

| Flag | Description |
|---|---|
| `--exam-id` | Exam ID to look up the registered layout for |
| `--register-layout PATH` | Register a layout JSON under its own exam_id (dev-only) |
| `--key PATH` | Path to the answer key JSON |
| `--student PATH` | Path to the student's filled sheet image |
| `--denoise-method` | `auto` / `none` / `median` / `nlmeans` (default: `nlmeans`) |
| `--binarize-method` | `otsu` / `adaptive` (default: `otsu`) |
| `--debug-images` | Save orientation-corrected and binarized images to `--outdir` |
| `--outdir` | Directory for debug images (default: `output`) |
| `--output-json PATH` | Write full results + summary to a JSON file |

## Known limitations (honest, not hidden)

- Answer key must currently be a JSON file (`{"1": "c", "2": "c", ...}`).
  Deriving the key from a scanned/marked image isn't implemented yet.
- The `layouts/` local-file store is a DEV-ONLY stand-in for the real
  paper-generator system's database. Once the two projects are integrated,
  swap `layout_store.get_stored_layout()`'s internals for a real DB/API
  call - nothing else in the pipeline needs to change.
- Page registration (perspective/homography correction for an
  off-angle photo) is not implemented - orientation correction only
  handles 90-degree-multiple rotation and horizontal flip.
- Mark detection threshold (`MARK_PIXEL_THRESHOLD` in `mark_detection.py`)
  is calibrated against our test sheets so far; needs validation against a
  wider set of real scanned/photographed sheets before production use.
