from __future__ import annotations

import pytest
from PIL import Image

from phd_boxprompt.qwen import (
    annotate_prompt_boxes,
    normalized_box_summary,
    parse_detections,
)
from phd_boxprompt.widget import box_widget

FENCED = """
Here you go.

```json
{"objects": [{"label": "match", "box_2d": [0, 0, 500, 250]},
             {"label": "match", "box_2d": [500, 500, 1000, 1000]}]}
```
"""


def test_parses_fenced_json_into_pixel_boxes():
    detections = parse_detections(FENCED, width=1000, height=400)
    assert [d.box for d in detections] == [(0, 0, 500, 100), (500, 200, 1000, 400)]


def test_parses_bare_json():
    detections = parse_detections('{"objects":[{"box_2d":[100,100,200,200]}]}', 200, 200)
    assert detections[0].box == (20, 20, 40, 40)


def test_swapped_and_out_of_range_coordinates_are_repaired():
    detections = parse_detections('{"objects":[{"box_2d":[900,-50,100,2000]}]}', 100, 100)
    assert detections[0].box == (10, 0, 90, 100)


def test_labels_are_kept_only_when_asked():
    payload = '{"objects":[{"label":"banderole","box_2d":[0,0,10,10]}]}'
    assert parse_detections(payload, 100, 100)[0].label == "match"
    assert parse_detections(payload, 100, 100, preserve_labels=True)[0].label == "banderole"


def test_empty_object_list_is_a_valid_no_match():
    assert parse_detections('{"objects": []}', 100, 100) == []


def test_unparseable_response_raises_instead_of_returning_empty():
    with pytest.raises(ValueError):
        parse_detections("I could not find anything useful.", 100, 100)


def test_normalized_summary_uses_thousandths():
    image = Image.new("RGB", (200, 400))
    boxes = [{"box": [0, 0, 100, 200], "kind": "positive"}]
    assert normalized_box_summary(image, boxes) == "positive: [0, 0, 500, 500]"


def test_annotation_does_not_mutate_the_source_image():
    image = Image.new("RGB", (64, 64), "white")
    before = image.tobytes()
    annotate_prompt_boxes(image, [{"box": [4, 4, 40, 40], "kind": "positive"}])
    assert image.tobytes() == before


def test_widget_reports_original_size_even_when_preview_is_downscaled():
    widget = box_widget(Image.new("RGB", (4000, 3000), "white"))
    assert (widget.img_width, widget.img_height) == (4000, 3000)
    assert widget.image.startswith("data:image/jpeg;base64,")
    assert widget.boxes == []


class _Message:
    """Stand-in for the SDK's message object."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_reasoning_read_from_plain_field():
    from phd_boxprompt.qwen import _extract_reasoning

    assert _extract_reasoning(_Message(reasoning="  I counted the bands.  ")) == (
        "I counted the bands."
    )


def test_reasoning_read_from_detail_blocks():
    from phd_boxprompt.qwen import _extract_reasoning

    message = _Message(
        reasoning=None,
        reasoning_details=[{"text": "first"}, {"summary": "second"}, {"text": "  "}],
    )
    assert _extract_reasoning(message) == "first\n\nsecond"


def test_reasoning_read_from_model_extra():
    from phd_boxprompt.qwen import _extract_reasoning

    message = _Message(model_extra={"reasoning": "hidden here"})
    assert _extract_reasoning(message) == "hidden here"


def test_reasoning_absent_is_empty_not_an_error():
    from phd_boxprompt.qwen import _extract_reasoning

    assert _extract_reasoning(_Message()) == ""


def test_result_still_unpacks_as_a_pair():
    from phd_boxprompt.qwen import BoxPromptResult, Detection

    result = BoxPromptResult([Detection((0, 0, 1, 1), "match")], "raw", "why", "m")
    detections, answer = result
    assert answer == "raw"
    assert detections[0].label == "match"


class _Delta:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Choice:
    def __init__(self, delta):
        self.delta = delta


class _Chunk:
    def __init__(self, choices):
        self.choices = choices


class _FakeCompletions:
    """Replays a canned stream and records the request it was given."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        assert kwargs.get("stream") is True
        return iter(self.chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.chat = _Delta(completions=_FakeCompletions(chunks))


def test_stream_accumulates_answer_and_reasoning_and_reports_progress():
    from phd_boxprompt.qwen import _consume_stream

    chunks = [
        _Chunk([]),  # keep-alive frame with no choices
        _Chunk([_Choice(_Delta(content=None, reasoning="I see "))]),
        _Chunk([_Choice(_Delta(content=None, reasoning="two bands."))]),
        _Chunk([_Choice(_Delta(content='{"objects":', reasoning=None))]),
        _Chunk([_Choice(_Delta(content=" []}", reasoning=None))]),
        _Chunk([_Delta(delta=None)]),  # malformed frame, must not crash
    ]
    seen = []
    client = _FakeClient(chunks)
    answer, reasoning = _consume_stream(client, {"model": "m"}, lambda r, a: seen.append((r, a)))

    assert answer == '{"objects": []}'
    assert reasoning == "I see two bands."
    assert len(seen) == 4
    assert seen[0] == ("I see ", "")
    assert seen[-1] == ("I see two bands.", '{"objects": []}')


def test_stream_reads_reasoning_from_model_extra_deltas():
    from phd_boxprompt.qwen import _delta_reasoning

    assert _delta_reasoning(_Delta(reasoning="direct")) == "direct"
    assert _delta_reasoning(_Delta(model_extra={"reasoning": "extra"})) == "extra"
    assert (
        _delta_reasoning(
            _Delta(model_extra={"reasoning_details": [{"text": "a"}, {"text": "b"}]})
        )
        == "ab"
    )
    assert _delta_reasoning(_Delta()) == ""


def test_standalone_notebook_is_up_to_date():
    """The molab file is generated; regenerating must be a no-op.

    If this fails, run: uv run python build_standalone.py
    """
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    import build_standalone

    generated = build_standalone.build()
    on_disk = build_standalone.OUTPUT_NOTEBOOK.read_text()
    assert generated == on_disk, "box_prompt_standalone.py is stale — rebuild it"


# --- SSE framing (the browser transport) ------------------------------------


def test_sse_splits_complete_frames_and_keeps_the_remainder():
    from phd_boxprompt.qwen import _split_sse_frames

    payloads, rest = _split_sse_frames('data: {"a":1}\n\ndata: {"b":2}\n\ndata: {"c"')
    assert payloads == ['{"a":1}', '{"b":2}']
    assert rest == 'data: {"c"'


def test_sse_frame_split_across_two_chunks_is_recovered():
    """A network chunk can end mid-frame; the leftover must carry over."""
    from phd_boxprompt.qwen import _split_sse_frames

    first, second = 'data: {"choices":[{"delta":{"con', 'tent":"hi"}}]}\n\n'
    payloads, rest = _split_sse_frames(first)
    assert payloads == []
    payloads, rest = _split_sse_frames(rest + second)
    assert payloads == ['{"choices":[{"delta":{"content":"hi"}}]}']
    assert rest == ""


def test_sse_ignores_keepalive_comments_and_crlf():
    from phd_boxprompt.qwen import _split_sse_frames

    buffer = ": OPENROUTER PROCESSING\r\n\r\ndata: {\"x\":1}\r\n\r\n"
    payloads, rest = _split_sse_frames(buffer)
    assert payloads == ['{"x":1}']
    assert rest == ""


def test_delta_parts_reads_content_and_reasoning():
    from phd_boxprompt.qwen import _mapping_delta_parts

    assert _mapping_delta_parts('{"choices":[{"delta":{"content":"abc"}}]}') == ("abc", "")
    assert _mapping_delta_parts('{"choices":[{"delta":{"reasoning":"why"}}]}') == ("", "why")
    assert _mapping_delta_parts(
        '{"choices":[{"delta":{"reasoning_details":[{"text":"a"},{"text":"b"}]}}]}'
    ) == ("", "ab")


def test_delta_parts_tolerates_done_empty_and_malformed_payloads():
    from phd_boxprompt.qwen import _mapping_delta_parts

    assert _mapping_delta_parts("[DONE]") == ("", "")
    assert _mapping_delta_parts("") == ("", "")
    assert _mapping_delta_parts("not json") == ("", "")
    assert _mapping_delta_parts('{"choices":[]}') == ("", "")
