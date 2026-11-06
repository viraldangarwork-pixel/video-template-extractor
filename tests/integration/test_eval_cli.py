import json
import shutil

import pytest
from typer.testing import CliRunner

from vtpl import cli

pytestmark = pytest.mark.slow


def test_eval_on_cards(tmp_path, cards_video):
    golden = tmp_path / "golden"
    golden.mkdir()
    shutil.copy(cards_video["path"], golden / "cards.mp4")
    (golden / "cards.expected.json").write_text(json.dumps({"cuts_ms": cards_video["cuts_ms"], "bpm": 120}))
    r = CliRunner().invoke(cli.app, ["eval", "--golden", str(golden), "--out", str(tmp_path / "work"),
                                     "--stages", "ingest,shots,audio,merge"])
    assert r.exit_code == 0, r.output
    assert "cuts_f1" in r.output
    report = json.loads((tmp_path / "work" / "eval_report.json").read_text())
    assert report["videos"]["cards"]["cuts"]["f1"] == 1.0 and report["videos"]["cards"]["bpm"]["ok"]
