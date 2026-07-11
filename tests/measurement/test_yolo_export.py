from pathlib import Path

from rbccps_od.data_management.yolo_export import output_dataset_yaml, yolo_line


def test_output_dataset_yaml_and_yolo_line(tmp_path: Path):
    output_dataset_yaml(tmp_path)
    content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
    assert "0: streetlight" in content
    line = yolo_line([0.0, 0.0, 10.0, 20.0], width=100, height=200)
    assert line.startswith("0 ")
