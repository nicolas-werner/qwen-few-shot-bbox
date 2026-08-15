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
```

Then paste your OpenRouter key into `.env` (already created, gitignored):

```
OPENROUTER_API_KEY=sk-or-...
```

Model defaults to `qwen/qwen3.8-max`. Override with `OPENROUTER_MODEL` — e.g.
`qwen/qwen3.8-2.4t-a95b` for the open-weight release, or any other vision model
on OpenRouter, which is the point: swapping one env var gives you a comparison.

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
- `temperature=0.0` by default, so repeated runs are as comparable as the
  provider allows.
- Whatever prompt string you use is part of the method. Record it with the
  results — it defines what counts as the object.
