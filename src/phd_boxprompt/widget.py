"""An anywidget for drawing positive/negative example boxes on an image.

Coordinates are always reported in *original image pixels*, independent of how
the image is scaled in the browser. Use it from marimo like this::

    import marimo as mo
    from phd_boxprompt.widget import BoxDrawWidget, box_widget

    widget = mo.ui.anywidget(box_widget("folio.jpg"))
    widget  # display it

    widget.value["boxes"]  # -> [{"box": [x1, y1, x2, y2], "kind": "positive"}, ...]
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import anywidget
import traitlets
from PIL import Image

__all__ = ["BoxDrawWidget", "box_widget"]

# Longest edge of the preview sent to the browser. Coordinates stay in original
# pixels; this only keeps the data URL from becoming enormous.
PREVIEW_MAX_EDGE = 1600

_ESM = r"""
const SVG_NS = "http://www.w3.org/2000/svg";

function render({ model, el }) {
  el.innerHTML = "";

  const root = document.createElement("div");
  root.className = "bp-root";

  // ---- toolbar -----------------------------------------------------------
  const bar = document.createElement("div");
  bar.className = "bp-bar";

  const makeButton = (text, extra) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = text;
    button.className = "bp-btn" + (extra ? " " + extra : "");
    return button;
  };

  const positiveButton = makeButton("＋ Positive", "bp-pos");
  const negativeButton = makeButton("－ Negative", "bp-neg");
  const undoButton = makeButton("Undo");
  const clearButton = makeButton("Clear");
  const counter = document.createElement("span");
  counter.className = "bp-count";

  bar.append(positiveButton, negativeButton, undoButton, clearButton, counter);

  // ---- stage -------------------------------------------------------------
  const stage = document.createElement("div");
  stage.className = "bp-stage";
  const img = document.createElement("img");
  img.draggable = false;
  img.alt = "Image to annotate";
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("preserveAspectRatio", "none");
  const canvas = document.createElement("canvas");
  canvas.setAttribute("aria-label", "Drag to draw a bounding box");
  stage.append(img, svg, canvas);

  const empty = document.createElement("div");
  empty.className = "bp-empty";
  empty.textContent = "No image loaded. Upload one above, then drag over it to draw a box.";

  root.append(bar, stage, empty);
  el.append(root);

  // ---- state -------------------------------------------------------------
  let kind = model.get("kind") || "positive";
  let start = null;
  let current = null;
  let drawing = false;

  const width = () => Number(model.get("img_width")) || 0;
  const height = () => Number(model.get("img_height")) || 0;
  const boxes = () => model.get("boxes") || [];

  const commit = (next) => {
    model.set("boxes", next);
    model.save_changes();
  };

  const setKind = (next) => {
    kind = next;
    model.set("kind", next);
    model.save_changes();
    positiveButton.classList.toggle("bp-active", next === "positive");
    negativeButton.classList.toggle("bp-active", next === "negative");
  };

  // ---- painting ----------------------------------------------------------
  const syncImage = () => {
    const source = model.get("image");
    const hasImage = Boolean(source) && width() > 0 && height() > 0;
    stage.style.display = hasImage ? "block" : "none";
    empty.style.display = hasImage ? "none" : "grid";
    if (!hasImage) return;
    img.src = source;
    svg.setAttribute("viewBox", `0 0 ${width()} ${height()}`);
    canvas.width = width();
    canvas.height = height();
  };

  const drawBoxes = () => {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const w = width();
    const h = height();
    if (!w || !h) return;
    const stroke = Math.max(2, Math.round(Math.min(w, h) / 200));
    const fontSize = Math.max(14, Math.round(Math.min(w, h) / 32));

    boxes().forEach((item, index) => {
      const [x1, y1, x2, y2] = item.box;
      const positive = item.kind === "positive";
      const color = positive ? "#16a34a" : "#dc2626";
      const label = `${positive ? "+" : "−"}${index + 1}`;

      const group = document.createElementNS(SVG_NS, "g");

      const rect = document.createElementNS(SVG_NS, "rect");
      rect.setAttribute("x", x1);
      rect.setAttribute("y", y1);
      rect.setAttribute("width", Math.max(0, x2 - x1));
      rect.setAttribute("height", Math.max(0, y2 - y1));
      rect.setAttribute("fill", "none");
      rect.setAttribute("stroke", color);
      rect.setAttribute("stroke-width", stroke);

      const tag = document.createElementNS(SVG_NS, "rect");
      tag.setAttribute("x", x1);
      tag.setAttribute("y", y1);
      tag.setAttribute("width", fontSize * 2.1);
      tag.setAttribute("height", fontSize + 6);
      tag.setAttribute("fill", color);

      const text = document.createElementNS(SVG_NS, "text");
      text.setAttribute("x", x1 + 5);
      text.setAttribute("y", y1 + fontSize);
      text.setAttribute("fill", "white");
      text.setAttribute("font-family", "sans-serif");
      text.setAttribute("font-size", fontSize);
      text.setAttribute("font-weight", "700");
      text.textContent = label;

      group.append(rect, tag, text);
      svg.append(group);
    });

    const positives = boxes().filter((b) => b.kind === "positive").length;
    const negatives = boxes().length - positives;
    counter.textContent = `${positives} positive · ${negatives} negative`;
  };

  const clearPreview = () => {
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
  };

  const drawPreview = () => {
    clearPreview();
    if (!start || !current) return;
    const context = canvas.getContext("2d");
    const stroke = Math.max(2, Math.round(Math.min(width(), height()) / 200));
    context.save();
    context.strokeStyle = kind === "positive" ? "#16a34a" : "#dc2626";
    context.lineWidth = stroke;
    context.setLineDash([stroke * 2, stroke * 1.5]);
    context.strokeRect(start.x, start.y, current.x - start.x, current.y - start.y);
    context.restore();
  };

  // ---- pointer handling --------------------------------------------------
  const pointAt = (event) => {
    const rect = canvas.getBoundingClientRect();
    const w = width();
    const h = height();
    return {
      x: Math.max(0, Math.min(w, Math.round(((event.clientX - rect.left) * w) / rect.width))),
      y: Math.max(0, Math.min(h, Math.round(((event.clientY - rect.top) * h) / rect.height))),
    };
  };

  canvas.addEventListener("pointerdown", (event) => {
    if (!width()) return;
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    drawing = true;
    start = pointAt(event);
    current = start;
    drawPreview();
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!drawing) return;
    current = pointAt(event);
    drawPreview();
  });

  const finish = (event) => {
    if (!drawing) return;
    drawing = false;
    current = pointAt(event);
    const x1 = Math.min(start.x, current.x);
    const y1 = Math.min(start.y, current.y);
    const x2 = Math.max(start.x, current.x);
    const y2 = Math.max(start.y, current.y);
    start = null;
    current = null;
    clearPreview();
    if (x2 - x1 < 4 || y2 - y1 < 4) return;
    commit([...boxes(), { box: [x1, y1, x2, y2], kind }]);
  };

  canvas.addEventListener("pointerup", finish);
  canvas.addEventListener("pointercancel", () => {
    drawing = false;
    start = null;
    current = null;
    clearPreview();
  });

  positiveButton.addEventListener("click", () => setKind("positive"));
  negativeButton.addEventListener("click", () => setKind("negative"));
  undoButton.addEventListener("click", () => commit(boxes().slice(0, -1)));
  clearButton.addEventListener("click", () => commit([]));

  model.on("change:boxes", drawBoxes);
  model.on("change:image", () => {
    syncImage();
    drawBoxes();
  });
  model.on("change:img_width", syncImage);
  model.on("change:img_height", syncImage);

  setKind(kind);
  syncImage();
  drawBoxes();
}

export default { render };
"""

_CSS = r"""
.bp-root { width: 100%; font-family: var(--marimo-font, system-ui, sans-serif); }
.bp-bar {
  display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center;
  margin-bottom: 0.5rem;
}
.bp-btn {
  padding: 0.28rem 0.7rem; border-radius: 6px; cursor: pointer;
  border: 1px solid #cbd5e1; background: #ffffff; color: #0f172a;
  font-size: 0.85rem; line-height: 1.3;
}
.bp-btn:hover { background: #f1f5f9; }
.bp-pos.bp-active { background: #16a34a; border-color: #16a34a; color: #ffffff; }
.bp-neg.bp-active { background: #dc2626; border-color: #dc2626; color: #ffffff; }
.bp-count { margin-left: auto; font-size: 0.8rem; color: #64748b; }
.bp-stage {
  position: relative; width: 100%; overflow: hidden;
  border: 1px solid #cbd5e1; border-radius: 8px; user-select: none;
  background: #f8fafc;
}
.bp-stage img { display: block; width: 100%; height: auto; pointer-events: none; }
.bp-stage svg, .bp-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
.bp-stage svg { z-index: 1; pointer-events: none; }
.bp-stage canvas { z-index: 2; cursor: crosshair; touch-action: none; }
.bp-empty {
  min-height: 220px; display: grid; place-items: center; padding: 2rem;
  border: 2px dashed #cbd5e1; border-radius: 8px; color: #64748b; text-align: center;
}
"""


class BoxDrawWidget(anywidget.AnyWidget):
    """Draw positive/negative example boxes over an image.

    Traits
    ------
    image : data URL of the (possibly downscaled) preview shown in the browser.
    img_width, img_height : size of the *original* image, in pixels.
    boxes : list of ``{"box": [x1, y1, x2, y2], "kind": "positive"|"negative"}``
        with coordinates in original-image pixels.
    kind : which kind the next drawn box will get.
    """

    _esm = _ESM
    _css = _CSS

    image = traitlets.Unicode("").tag(sync=True)
    img_width = traitlets.Int(0).tag(sync=True)
    img_height = traitlets.Int(0).tag(sync=True)
    boxes = traitlets.List(traitlets.Dict()).tag(sync=True)
    kind = traitlets.Unicode("positive").tag(sync=True)


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def box_widget(source: str | Path | bytes | Image.Image) -> BoxDrawWidget:
    """Build a :class:`BoxDrawWidget` from a path, raw bytes, or a PIL image."""
    if isinstance(source, Image.Image):
        image = source
    elif isinstance(source, bytes):
        image = Image.open(io.BytesIO(source))
    else:
        image = Image.open(Path(source))

    image = image.convert("RGB")
    original_width, original_height = image.size

    preview = image.copy()
    preview.thumbnail((PREVIEW_MAX_EDGE, PREVIEW_MAX_EDGE), Image.Resampling.LANCZOS)

    return BoxDrawWidget(
        image=_data_url(preview),
        img_width=original_width,
        img_height=original_height,
        boxes=[],
    )
