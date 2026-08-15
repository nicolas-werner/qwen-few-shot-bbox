"""Few-shot box prompting against Qwen3.8-Max, served through OpenRouter.

The trick borrowed from the Gradio reference app: the example boxes are *drawn
onto* the image before sending, and additionally passed as normalized
coordinates in the text prompt. The model sees the exemplars both ways.

Any vision model on OpenRouter works — set ``OPENROUTER_MODEL`` or pass
``model=``. Useful slugs:

``qwen/qwen3.8-max``          proprietary hosted version (default)
``qwen/qwen3.8-2.4t-a95b``    the open-weight MoE release
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

__all__ = [
    "BoxPromptResult",
    "Detection",
    "annotate_prompt_boxes",
    "detect_similar",
    "draw_detections",
    "parse_detections",
]

DEFAULT_MODEL = "qwen/qwen3.8-max"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

POSITIVE_COLOR = "#16a34a"
NEGATIVE_COLOR = "#dc2626"
MATCH_COLOR = "#2563eb"


@dataclass(frozen=True)
class Detection:
    """One returned box, in original-image pixels."""

    box: tuple[int, int, int, int]
    label: str


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def annotate_prompt_boxes(
    image: Image.Image,
    boxes: list[dict[str, Any]],
) -> Image.Image:
    """Return a copy of ``image`` with the exemplar boxes painted on it."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    line_width = max(3, round(min(canvas.size) / 180))

    for index, annotation in enumerate(boxes, start=1):
        x1, y1, x2, y2 = (int(v) for v in annotation["box"])
        positive = annotation.get("kind") == "positive"
        color = POSITIVE_COLOR if positive else NEGATIVE_COLOR
        label = f" {'+' if positive else '-'}{index} "
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        text_box = draw.textbbox((x1, y1), label)
        draw.rectangle(text_box, fill=color)
        draw.text((x1, y1), label, fill="white")

    return canvas


def draw_detections(
    image: Image.Image,
    prompt_boxes: list[dict[str, Any]],
    detections: list[Detection],
) -> Image.Image:
    """Paint exemplars (green/red) and model matches (blue) onto a copy."""
    canvas = annotate_prompt_boxes(image, prompt_boxes)
    draw = ImageDraw.Draw(canvas, "RGBA")
    line_width = max(2, round(min(canvas.size) / 260))

    for detection in detections:
        draw.rectangle(detection.box, outline=MATCH_COLOR, width=line_width)
        draw.rectangle(detection.box, fill=(37, 99, 235, 40))

    return canvas


def normalized_box_summary(image: Image.Image, boxes: list[dict[str, Any]]) -> str:
    """Exemplar boxes as 0-1000 normalized XYXY, for the text half of the prompt."""
    width, height = image.size
    parts = []
    for annotation in boxes:
        x1, y1, x2, y2 = annotation["box"]
        normalized = [
            round(1000 * x1 / width),
            round(1000 * y1 / height),
            round(1000 * x2 / width),
            round(1000 * y2 / height),
        ]
        parts.append(f"{annotation.get('kind', 'positive')}: {normalized}")
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# response parsing
# --------------------------------------------------------------------------- #


def parse_detections(
    answer: str,
    width: int,
    height: int,
    preserve_labels: bool = False,
) -> list[Detection]:
    """Pull ``{"objects": [{"box_2d": [...]}]}`` out of the model's text.

    Raises ``ValueError`` if nothing parseable is found. It deliberately does not
    return an empty list on failure, so a parse error is never silently read as
    "the model found nothing".
    """
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", answer, flags=re.IGNORECASE)
    candidate = fenced.group(1) if fenced else answer

    object_match = re.search(r"\{[\s\S]*\}", candidate)
    list_match = re.search(r"\[[\s\S]*\]", candidate)
    if not (object_match or list_match):
        raise ValueError("no JSON object or array found in the model response")

    payload = json.loads((object_match or list_match).group(0))
    objects = payload.get("objects", []) if isinstance(payload, dict) else payload

    detections: list[Detection] = []
    for item in objects:
        coordinates = item.get("box_2d") or item.get("bbox") or item.get("box")
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in coordinates)
        x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
        y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
        pixel_box = (
            round(x1 * width / 1000),
            round(y1 * height / 1000),
            round(x2 * width / 1000),
            round(y2 * height / 1000),
        )
        label = str(item.get("label") or "match")[:80] if preserve_labels else "match"
        detections.append(Detection(box=pixel_box, label=label))

    return detections


# --------------------------------------------------------------------------- #
# the API call
# --------------------------------------------------------------------------- #


def _client(timeout: float = 180.0):
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Paste your key into .env "
            "(copy .env.example if .env is missing), then restart the notebook."
        )
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        timeout=timeout,
        max_retries=2,
    )


def _extra_headers() -> dict[str, str]:
    """Optional OpenRouter attribution headers. Both are safe to leave unset."""
    headers = {}
    if referer := os.getenv("OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = referer
    if title := os.getenv("OPENROUTER_SITE_NAME"):
        headers["X-OpenRouter-Title"] = title
    return headers


PROMPT_TEMPLATE = """
The image contains visual box prompts drawn by the user. Green boxes marked with + are \
positive examples of the object to find. Red boxes marked with - are negative examples \
that must be ignored.

The user's instruction is: {instruction}

Find every other unboxed object in this same image that matches the positive examples \
while respecting the negative examples. Return only valid JSON in this exact shape:

{{"objects": [{{"label": "match", "box_2d": [x_min, y_min, x_max, y_max]}}]}}

Use XYXY coordinates normalized to integers from 0 to 1000. Do not return the already \
marked prompt boxes. If there are no additional matches, return {{"objects": []}}.

For reference, the prompt boxes in normalized XYXY coordinates are: {box_summary}
""".strip()


def _extract_reasoning(message: Any) -> str:
    """Pull the reasoning trace out of an OpenRouter message, if the provider sent one.

    OpenRouter exposes it as ``message.reasoning`` (plain text) and/or
    ``message.reasoning_details`` (a list of blocks). Neither is guaranteed —
    some providers strip it, some models have none. Returns "" when absent.
    """
    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    details = getattr(message, "reasoning_details", None)
    if not details:
        # Non-standard fields survive on model_extra rather than as attributes.
        extra = getattr(message, "model_extra", None) or {}
        if isinstance(extra.get("reasoning"), str):
            return str(extra["reasoning"]).strip()
        details = extra.get("reasoning_details")

    blocks: list[str] = []
    for block in details or []:
        if isinstance(block, dict):
            text = block.get("text") or block.get("summary") or block.get("data")
        else:
            text = getattr(block, "text", None) or getattr(block, "summary", None)
        if isinstance(text, str) and text.strip():
            blocks.append(text.strip())
    return "\n\n".join(blocks)


@dataclass(frozen=True)
class BoxPromptResult:
    """One box-prompt run: what came back, and how the model got there."""

    detections: list[Detection]
    answer: str
    reasoning: str = ""
    model: str = ""

    def __iter__(self):
        """Backwards-compatible ``detections, answer = detect_similar(...)``."""
        return iter((self.detections, self.answer))


def detect_similar(
    image: Image.Image,
    boxes: list[dict[str, Any]],
    instruction: str = "",
    *,
    model: str | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    reasoning: bool | str = True,
) -> BoxPromptResult:
    """Run one few-shot box prompt.

    ``boxes`` is the widget's value: ``[{"box": [x1, y1, x2, y2], "kind": ...}]``
    in original-image pixels. At least one positive box is required.

    ``reasoning`` asks OpenRouter for the model's reasoning trace. Pass ``True``
    to simply enable it, an effort level (``"low"``/``"medium"``/``"high"``) to
    tune it, or ``False`` to skip. Providers that do not support it are handled
    by retrying once without the parameter, so this never hard-fails a run.
    """
    if not any(b.get("kind") == "positive" for b in boxes):
        raise ValueError("draw at least one positive box before running detection")

    instruction = instruction.strip() or (
        "Find every other unboxed object that visually matches the positive examples."
    )
    prompt = PROMPT_TEMPLATE.format(
        instruction=instruction,
        box_summary=normalized_box_summary(image, boxes),
    )
    annotated = annotate_prompt_boxes(image, boxes)
    resolved_model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

    request: dict[str, Any] = {
        "model": resolved_model,
        "extra_headers": _extra_headers(),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": _data_url(annotated)},
                    },
                ],
            }
        ],
        "max_tokens": int(max_tokens),
        "temperature": temperature,
    }

    if reasoning:
        request["extra_body"] = {
            "reasoning": (
                {"effort": reasoning} if isinstance(reasoning, str) else {"enabled": True}
            )
        }

    client = _client()
    try:
        completion = client.chat.completions.create(**request)
    except Exception:
        if not reasoning:
            raise
        # The provider rejected the reasoning parameter — rerun plainly.
        request.pop("extra_body", None)
        completion = client.chat.completions.create(**request)

    message = completion.choices[0].message
    answer = str(message.content or "").strip()
    if not answer:
        raise RuntimeError("the model returned no text; try raising max_tokens")

    width, height = image.size
    return BoxPromptResult(
        detections=parse_detections(answer, width, height),
        answer=answer,
        reasoning=_extract_reasoning(message),
        model=resolved_model,
    )
