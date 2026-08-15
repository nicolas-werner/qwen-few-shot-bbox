# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "anywidget>=0.9.13",
#     "pillow>=10.4.0",
#     "traitlets>=5.14.3",
#     "openai>=1.60.0; sys_platform != 'emscripten'",
#     "python-dotenv>=1.0.1; sys_platform != 'emscripten'",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import json
    import os
    import time

    import marimo as mo
    from PIL import Image

    # <<<INLINE-IMPORTS>>>
    from phd_boxprompt.qwen import (
        adetect_similar,
        detect_similar,
        draw_detections,
        running_in_browser,
    )
    from phd_boxprompt.widget import box_widget
    # <<<END-INLINE-IMPORTS>>>

    # A .env file is a convenience, not a requirement — and python-dotenv is not
    # installed in the browser build.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    return (
        Image,
        adetect_similar,
        box_widget,
        detect_similar,
        draw_detections,
        io,
        json,
        mo,
        os,
        running_in_browser,
        time,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Box prompting

        Drop an image below, draw one or two **example boxes** on it, and ask the
        model to find the rest. Coordinates are reported in original-image pixels.
        """
    )
    return


@app.cell
def _(mo, os):
    api_key_input = mo.ui.text(
        label="OpenRouter API key",
        placeholder="sk-or-…",
        kind="password",
        full_width=True,
    )
    key_from_env = bool((os.getenv("OPENROUTER_API_KEY") or "").strip())
    return api_key_input, key_from_env


@app.cell
def _(api_key_input, key_from_env, mo):
    if api_key_input.value.strip():
        key_status = "Using the key you pasted below."
    elif key_from_env:
        key_status = "Using `OPENROUTER_API_KEY` from the environment — no need to paste one."
    else:
        key_status = (
            "No key found. Paste one below — get a free key at "
            "[openrouter.ai/keys](https://openrouter.ai/keys). "
            "It stays in this browser tab and is never written to disk."
        )

    mo.vstack([mo.callout(mo.md(key_status), kind="info"), api_key_input])
    return (key_status,)


@app.cell
def _(mo):
    upload = mo.ui.file(
        kind="area",
        filetypes=[".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"],
        multiple=False,
        label="Drop a folio here, or click to browse",
        max_size=50_000_000,
    )
    upload
    return (upload,)


@app.cell
def _(Image, io, mo, upload):
    mo.stop(not upload.value, mo.md("*Waiting for an image.*"))

    image = Image.open(io.BytesIO(upload.contents(0))).convert("RGB")
    mo.md(f"**{upload.name(0)}** — {image.width} × {image.height} px")
    return (image,)


@app.cell
def _(box_widget, image, mo):
    widget = mo.ui.anywidget(box_widget(image))
    widget
    return (widget,)


@app.cell
def _(mo, widget):
    boxes = widget.value["boxes"]
    positives = [b for b in boxes if b["kind"] == "positive"]

    instruction = mo.ui.text_area(
        label="What should the model find?",
        placeholder="e.g. Find every other speech scroll. Ignore the rubricated caption strip.",
        rows=3,
        full_width=True,
    )
    run = mo.ui.run_button(label="Run detection", disabled=not positives)

    mo.vstack(
        [
            mo.md(
                f"`{len(positives)}` positive · `{len(boxes) - len(positives)}` negative"
                + ("" if positives else " — draw at least one green box to enable the run.")
            ),
            instruction,
            run,
        ]
    )
    return boxes, instruction, run


@app.cell
async def _(
    adetect_similar,
    api_key_input,
    boxes,
    detect_similar,
    image,
    instruction,
    mo,
    run,
    running_in_browser,
    time,
):
    mo.stop(not run.value, mo.md("*Draw your boxes, then click **Run detection**.*"))

    # Every replace() re-serialises and broadcasts the whole block, so cap the
    # refresh rate rather than repainting on each token.
    last_paint = [0.0]
    REFRESH_SECONDS = 0.1

    def show_progress(reasoning_so_far: str, answer_so_far: str) -> None:
        now = time.monotonic()
        if now - last_paint[0] < REFRESH_SECONDS:
            return
        last_paint[0] = now
        panels = []
        if reasoning_so_far:
            panels.append(mo.md(f"**Reasoning…**\n\n{reasoning_so_far}"))
        panels.append(mo.md(f"**Response…**\n\n```json\n{answer_so_far or '▌'}\n```"))
        mo.output.replace(mo.vstack(panels))

    key = api_key_input.value.strip() or None

    if running_in_browser():
        # Pyodide has no sockets, so this reads the response stream by hand.
        mo.output.replace(mo.md("**Asking the model…**"))
        result = await adetect_similar(
            image, boxes, instruction.value, api_key=key, on_chunk=show_progress
        )
    else:
        result = detect_similar(
            image, boxes, instruction.value, api_key=key, on_chunk=show_progress
        )

    mo.output.replace(
        mo.md(f"Done — **{len(result.detections)}** match(es) from `{result.model}`.")
    )
    return (result,)


@app.cell
def _(boxes, draw_detections, image, mo, result):
    mo.vstack(
        [
            mo.md(f"### {len(result.detections)} match(es)"),
            mo.image(draw_detections(image, boxes, result.detections), width=900),
        ]
    )
    return


@app.cell
def _(mo, result):
    mo.md(
        f"## Reasoning\n\n{result.reasoning}"
        if result.reasoning
        else "## Reasoning\n\n*This provider returned no reasoning trace for the run.*"
    )
    return


@app.cell
def _(boxes, instruction, json, mo, result):
    record = {
        "model": result.model,
        "instruction": instruction.value,
        "prompt_boxes": boxes,
        "detections": [{"box": list(d.box), "label": d.label} for d in result.detections],
        "reasoning": result.reasoning,
    }

    mo.accordion(
        {
            "Result as JSON (copy this into your notes)": mo.md(
                f"```json\n{json.dumps(record, indent=2)}\n```"
            ),
            "Raw model response": mo.md(f"```json\n{result.answer}\n```"),
        }
    )
    return


if __name__ == "__main__":
    app.run()
