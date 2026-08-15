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

A key is needed to run detection. Either paste it into the field at the top of
the notebook, or put it in `.env` (gitignored) so you don't retype it:

```
OPENROUTER_API_KEY=sk-or-...
```

The `.env` is optional — the notebook works without one, and the browser/molab
copy has no `.env` at all.

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
notebooks/box_prompt.py              the notebook — source of truth
notebooks/box_prompt_standalone.py   generated single file for molab (do not edit)
build_standalone.py                  regenerates the above
src/phd_boxprompt/widget.py          BoxDrawWidget — anywidget canvas, drag to draw
src/phd_boxprompt/qwen.py            OpenRouter call, prompt, response parsing
tests/                               parsing + streaming tests (no network)
```

## Sharing with molab

[molab](https://docs.marimo.io/guides/molab/) runs real CPython in the cloud, so
streaming works there exactly as it does locally. It is single-file, though: it
syncs one notebook from a GitHub URL and cannot import `phd_boxprompt`. That is
what `notebooks/box_prompt_standalone.py` is for — the same notebook with the package
inlined into its first cell.

```bash
uv run python build_standalone.py     # after any change to the notebook or package
```

Then push and point molab at the GitHub URL of
`notebooks/box_prompt_standalone.py`. A test fails if the
generated file drifts from its sources, so it cannot go stale silently.

Two things to tell whoever you share it with:

- They need their own OpenRouter key — there is a field at the top of the
  notebook. It is typed in at runtime and never written into the file.
- molab notebooks are **public but undiscoverable**: anyone with the link can
  read the code. Nothing secret lives in it, but it is not private.

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

## Streaming

`detect_similar(..., on_chunk=fn)` streams: `fn(reasoning_so_far, answer_so_far)`
is called after every delta. The notebook uses it with `mo.output.replace` so the
reasoning and the JSON appear as they are generated. Omit `on_chunk` and the call
is a single blocking request instead.

Keep-alive and usage frames (which carry no choices) are skipped, so the
progress callback only fires on real content.

## Reasoning trace

`detect_similar(..., reasoning=True)` asks OpenRouter for the model's reasoning
and returns it on `BoxPromptResult.reasoning`; the notebook shows it under the
result image and includes it in the JSON record. Pass an effort level
(`"low"`/`"medium"`/`"high"`) instead of `True` to tune it, or `False` to skip.

Providers that reject the parameter are retried once without it, and a model
that simply returns no trace yields `""` — never an error. So an empty
reasoning field means *the provider sent none*, not *the model did not think*.
That distinction matters if you plan to cite the trace as evidence.
