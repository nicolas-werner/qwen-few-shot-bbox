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
