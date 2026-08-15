import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import json

    import marimo as mo
    from dotenv import load_dotenv
    from PIL import Image

    from phd_boxprompt.qwen import detect_similar, draw_detections
    from phd_boxprompt.widget import box_widget

    load_dotenv()
    return (
        Image,
        box_widget,
        detect_similar,
        draw_detections,
        io,
        json,
        mo,
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
def _(boxes, detect_similar, image, instruction, mo, run):
    mo.stop(not run.value, mo.md("*Draw your boxes, then click **Run detection**.*"))

    with mo.status.spinner(title="Asking the model…"):
        detections, raw_answer = detect_similar(image, boxes, instruction.value)
    return detections, raw_answer


@app.cell
def _(boxes, detections, draw_detections, image, mo):
    mo.vstack(
        [
            mo.md(f"### {len(detections)} match(es)"),
            mo.image(draw_detections(image, boxes, detections), width=900),
        ]
    )
    return


@app.cell
def _(boxes, detections, instruction, json, mo, raw_answer):
    record = {
        "instruction": instruction.value,
        "prompt_boxes": boxes,
        "detections": [{"box": list(d.box), "label": d.label} for d in detections],
    }

    mo.accordion(
        {
            "Result as JSON (copy this into your notes)": mo.md(
                f"```json\n{json.dumps(record, indent=2)}\n```"
            ),
            "Raw model response": mo.md(f"```json\n{raw_answer}\n```"),
        }
    )
    return


if __name__ == "__main__":
    app.run()
