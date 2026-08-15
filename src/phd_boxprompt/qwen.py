"""Few-shot box prompting against Qwen3.8-Max, served through OpenRouter.

The trick borrowed from the Gradio reference app: the example boxes are *drawn
onto* the image before sending, and additionally passed as normalized
coordinates in the text prompt. The model sees the exemplars both ways.

Any vision model on OpenRouter works — set ``OPENROUTER_MODEL`` or pass
``model=``. Useful slugs:

``qwen/qwen3.8-max``          proprietary hosted version (default)
``qwen/qwen3.8-2.4t-a95b``    the open-weight MoE release

Two transports, picked automatically:

* CPython — the ``openai`` SDK, with token-by-token streaming.
* Pyodide (a WASM-exported notebook in the browser) — ``pyodide.http.pyfetch``,
  since the SDK needs sockets the browser does not give us. Use the ``async``
  twin :func:`adetect_similar` there. No streaming on that path.
"""

from __future__ import annotations

import base64
import codecs
import io
import json
import os
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

__all__ = [
    "BoxPromptResult",
    "Detection",
    "adetect_similar",
    "adetect_similar",
    "annotate_prompt_boxes",
    "detect_similar",
    "draw_detections",
    "parse_detections",
    "resolve_api_key",
    "running_in_browser",
]

DEFAULT_MODEL = "qwen/qwen3.8-max"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

POSITIVE_COLOR = "#16a34a"
NEGATIVE_COLOR = "#dc2626"
MATCH_COLOR = "#2563eb"

MISSING_KEY_MESSAGE = (
    "No OpenRouter API key. Paste one into the key field in the notebook, or set "
    "OPENROUTER_API_KEY in the environment (or in a .env file). "
    "Keys are free to create at https://openrouter.ai/keys"
)


def running_in_browser() -> bool:
    """True when executing inside Pyodide, i.e. the WASM build."""
    return sys.platform == "emscripten"


@dataclass(frozen=True)
class Detection:
    """One returned box, in original-image pixels."""

    box: tuple[int, int, int, int]
    label: str


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


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def annotate_prompt_boxes(image: Image.Image, boxes: list[dict[str, Any]]) -> Image.Image:
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


def _reasoning_from_blocks(details: Any) -> str:
    blocks: list[str] = []
    for block in details or []:
        if isinstance(block, dict):
            text = block.get("text") or block.get("summary") or block.get("data")
        else:
            text = getattr(block, "text", None) or getattr(block, "summary", None)
        if isinstance(text, str) and text.strip():
            blocks.append(text.strip())
    return "\n\n".join(blocks)


def _extract_reasoning(message: Any) -> str:
    """Pull the reasoning trace out of a message, dict or SDK object alike.

    OpenRouter exposes it as ``reasoning`` (plain text) and/or
    ``reasoning_details`` (a list of blocks). Neither is guaranteed — some
    providers strip it, some models have none. Returns "" when absent.
    """
    if isinstance(message, Mapping):
        text = message.get("reasoning")
        if isinstance(text, str) and text.strip():
            return text.strip()
        return _reasoning_from_blocks(message.get("reasoning_details"))

    text = getattr(message, "reasoning", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    details = getattr(message, "reasoning_details", None)
    if not details:
        # Non-standard fields survive on model_extra rather than as attributes.
        extra = getattr(message, "model_extra", None) or {}
        if isinstance(extra.get("reasoning"), str):
            return str(extra["reasoning"]).strip()
        details = extra.get("reasoning_details")
    return _reasoning_from_blocks(details)


def _delta_reasoning(delta: Any) -> str:
    """Reasoning text carried by one streamed delta, if any."""
    text = getattr(delta, "reasoning", None)
    if isinstance(text, str):
        return text
    extra = getattr(delta, "model_extra", None) or {}
    if isinstance(extra.get("reasoning"), str):
        return str(extra["reasoning"])
    blocks = extra.get("reasoning_details") or []
    return "".join(b.get("text", "") for b in blocks if isinstance(b, dict))


def _split_sse_frames(buffer: str) -> tuple[list[str], str]:
    """Split an SSE buffer into complete ``data:`` payloads plus the remainder.

    A network chunk can end mid-frame, so whatever follows the last blank line
    is handed back to be prepended to the next chunk. Comment lines (OpenRouter
    sends ``: OPENROUTER PROCESSING`` as a keep-alive) are dropped.
    """
    normalized = buffer.replace("\r\n", "\n")
    payloads: list[str] = []

    while "\n\n" in normalized:
        frame, normalized = normalized.split("\n\n", 1)
        for line in frame.split("\n"):
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                payloads.append(line[len("data:") :].strip())

    return payloads, normalized


def _mapping_delta_parts(payload: str) -> tuple[str, str]:
    """``(answer, reasoning)`` carried by one SSE payload. ``("", "")`` if none."""
    if not payload or payload == "[DONE]":
        return "", ""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return "", ""

    choices = data.get("choices") or []
    if not choices:
        return "", ""
    delta = choices[0].get("delta") or {}

    answer = delta.get("content") or ""
    reasoning = delta.get("reasoning") or ""
    if not reasoning:
        reasoning = "".join(
            block.get("text", "")
            for block in (delta.get("reasoning_details") or [])
            if isinstance(block, dict)
        )
    return str(answer), str(reasoning)


# --------------------------------------------------------------------------- #
# request construction
# --------------------------------------------------------------------------- #


def resolve_api_key(api_key: str | None = None) -> str:
    """Explicit key wins, then the environment. Raises with a helpful message."""
    key = (api_key or "").strip() or (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(MISSING_KEY_MESSAGE)
    return key


def _attribution_headers() -> dict[str, str]:
    """Optional OpenRouter attribution — cosmetic, only affects their rankings."""
    headers = {}
    if referer := os.getenv("OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = referer
    if title := os.getenv("OPENROUTER_SITE_NAME"):
        headers["X-OpenRouter-Title"] = title
    return headers


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **_attribution_headers(),
    }


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **_attribution_headers(),
    }


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


def build_payload(
    image: Image.Image,
    boxes: list[dict[str, Any]],
    instruction: str = "",
    *,
    model: str | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    reasoning: bool | str = True,
) -> dict[str, Any]:
    """The JSON body for one box-prompt request. Shared by both transports."""
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

    payload: dict[str, Any] = {
        "model": model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _data_url(annotated)}},
                ],
            }
        ],
        "max_tokens": int(max_tokens),
        "temperature": temperature,
    }
    if reasoning:
        payload["reasoning"] = (
            {"effort": reasoning} if isinstance(reasoning, str) else {"enabled": True}
        )
    return payload


def _finish(payload: dict[str, Any], answer: str, reasoning: str, size) -> BoxPromptResult:
    answer = answer.strip()
    if not answer:
        raise RuntimeError("the model returned no text; try raising max_tokens")
    width, height = size
    return BoxPromptResult(
        detections=parse_detections(answer, width, height),
        answer=answer,
        reasoning=reasoning.strip(),
        model=str(payload.get("model", "")),
    )


# --------------------------------------------------------------------------- #
# transport: CPython, via the openai SDK
# --------------------------------------------------------------------------- #


def _consume_stream(
    client: Any,
    request: dict[str, Any],
    on_chunk: Callable[[str, str], None],
) -> tuple[str, str]:
    """Stream a completion, reporting progress. Returns ``(answer, reasoning)``."""
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []

    stream: Iterator[Any] = client.chat.completions.create(**request, stream=True)
    for chunk in stream:
        # OpenRouter interleaves keep-alive and usage frames with no choices.
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        changed = False
        if content := getattr(delta, "content", None):
            answer_parts.append(str(content))
            changed = True
        if thought := _delta_reasoning(delta):
            reasoning_parts.append(thought)
            changed = True

        if changed:
            on_chunk("".join(reasoning_parts), "".join(answer_parts))

    return "".join(answer_parts), "".join(reasoning_parts)


def detect_similar(
    image: Image.Image,
    boxes: list[dict[str, Any]],
    instruction: str = "",
    *,
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    reasoning: bool | str = True,
    on_chunk: Callable[[str, str], None] | None = None,
) -> BoxPromptResult:
    """Run one few-shot box prompt (CPython transport).

    ``boxes`` is the widget's value: ``[{"box": [x1, y1, x2, y2], "kind": ...}]``
    in original-image pixels. At least one positive box is required.

    ``api_key`` overrides ``OPENROUTER_API_KEY``; either may supply the key.

    ``reasoning`` asks OpenRouter for the model's reasoning trace. Pass ``True``
    to enable it, an effort level (``"low"``/``"medium"``/``"high"``) to tune it,
    or ``False`` to skip. Providers that do not support it are handled by
    retrying once without the parameter, so this never hard-fails a run.

    Pass ``on_chunk`` to stream. It is called with the accumulated
    ``(reasoning, answer)`` after every delta.
    """
    from openai import OpenAI

    payload = build_payload(
        image,
        boxes,
        instruction,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning=reasoning,
    )
    client = OpenAI(
        api_key=resolve_api_key(api_key),
        base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        timeout=180.0,
        max_retries=2,
    )

    request = {k: v for k, v in payload.items() if k != "reasoning"}
    request["extra_headers"] = _attribution_headers()
    if "reasoning" in payload:
        request["extra_body"] = {"reasoning": payload["reasoning"]}

    def call() -> tuple[str, str]:
        if on_chunk is None:
            completion = client.chat.completions.create(**request)
            message = completion.choices[0].message
            return str(message.content or ""), _extract_reasoning(message)
        return _consume_stream(client, request, on_chunk)

    try:
        answer, trace = call()
    except Exception:
        if "extra_body" not in request:
            raise
        # The provider rejected the reasoning parameter — rerun plainly.
        request.pop("extra_body", None)
        answer, trace = call()

    return _finish(payload, answer, trace, image.size)


# --------------------------------------------------------------------------- #
# transport: Pyodide, via the browser's fetch
# --------------------------------------------------------------------------- #


async def _consume_pyfetch_stream(
    response: Any,
    on_chunk: Callable[[str, str], None],
) -> tuple[str, str]:
    """Read an SSE response body chunk by chunk. Returns ``(answer, reasoning)``.

    Bytes are decoded incrementally, because a multi-byte character can be split
    across two network chunks; the same is true of SSE frames, which is what the
    leftover buffer is for.
    """
    reader = response.js_response.body.getReader()
    decoder = codecs.getincrementaldecoder("utf-8")()

    buffer = ""
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []

    while True:
        chunk = await reader.read()
        if chunk.done:
            break

        buffer += decoder.decode(bytes(chunk.value.to_py()))
        payloads, buffer = _split_sse_frames(buffer)

        changed = False
        for payload in payloads:
            if payload == "[DONE]":
                continue
            answer, thought = _mapping_delta_parts(payload)
            if answer:
                answer_parts.append(answer)
                changed = True
            if thought:
                reasoning_parts.append(thought)
                changed = True

        if changed:
            on_chunk("".join(reasoning_parts), "".join(answer_parts))

    return "".join(answer_parts), "".join(reasoning_parts)


async def adetect_similar(
    image: Image.Image,
    boxes: list[dict[str, Any]],
    instruction: str = "",
    *,
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    reasoning: bool | str = True,
    on_chunk: Callable[[str, str], None] | None = None,
) -> BoxPromptResult:
    """Run one few-shot box prompt from a browser (Pyodide) notebook.

    Uses ``pyodide.http.pyfetch`` because the ``openai`` SDK needs sockets that
    WASM does not provide. OpenRouter serves permissive CORS headers, so the
    request works directly from the page.

    Pass ``on_chunk`` to stream, exactly as with :func:`detect_similar`. The
    browser hands us a ``ReadableStream`` rather than an SSE client, so the
    frames are parsed here.
    """
    from pyodide.http import pyfetch  # only exists inside Pyodide

    payload = build_payload(
        image,
        boxes,
        instruction,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning=reasoning,
    )
    base = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

    streaming = on_chunk is not None
    if streaming:
        payload["stream"] = True

    response = await pyfetch(
        f"{base}/chat/completions",
        method="POST",
        headers=_headers(resolve_api_key(api_key)),
        body=json.dumps(payload),
    )
    if response.status != 200:
        detail = (await response.string())[:400]
        raise RuntimeError(f"OpenRouter returned HTTP {response.status}: {detail}")

    if streaming:
        answer, trace = await _consume_pyfetch_stream(response, on_chunk)
    else:
        data = await response.json()
        if error := data.get("error"):
            raise RuntimeError(f"OpenRouter error: {error}")
        message = data["choices"][0]["message"]
        answer, trace = str(message.get("content") or ""), _extract_reasoning(message)

    payload.pop("stream", None)
    return _finish(payload, answer, trace, image.size)
