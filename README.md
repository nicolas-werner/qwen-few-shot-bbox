# phd — box-prompt experiments

A marimo notebook for few-shot **box prompting**: drag an image in, draw one or
two example boxes on it, and ask a vision model to find the rest.

Built to probe what exemplar prompting does on manuscript folios — where the
objects (speech scrolls, banderoles, wheel segments) are curved, overlapping, and
not box-shaped. Coordinates are always reported in original-image pixels, so a
result can be compared against a real annotation.

## Setup

```bash
uv sync
cp .env.example .env      # then paste your DASHSCOPE_API_KEY
```

## Run

```bash
uv run marimo edit notebooks/box_prompt.py
```

Drop an image into the upload area, pick **Positive** or **Negative**, drag over
the image to draw boxes, write what the model should look for, then hit *Run
detection*. The API is only called on that click.

## Layout

```
notebooks/box_prompt.py       the notebook
src/phd_boxprompt/widget.py   BoxDrawWidget — anywidget canvas, drag to draw
src/phd_boxprompt/qwen.py     DashScope call, prompt template, response parsing
tests/                        parsing tests (no network)
```

## Notes

- The exemplars are sent **twice**: painted onto the image, and as normalized
  0–1000 coordinates in the text prompt. That is what the Qwen demo does, and it
  measurably beats coordinates alone.
- `parse_detections` raises on unparseable output rather than returning `[]`, so
  a broken response is never silently read as "found nothing".
- Model and endpoint are overridable via `QWEN_MODEL` and `QWEN_BASE_URL`.
- Whatever prompt string you use is part of the method. Record it with the
  results — it defines what counts as the object.
