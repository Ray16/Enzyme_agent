"""Publication lookup still resolves figures after co-locating both case studies."""
import pytest
from structural_modeling_agent.publish_figures import figure_manifest
from structural_modeling_agent import integration
import hashlib
import json


def test_two_studies_resolve_separate_masters_and_reject_duplicates(tmp_path):
    synthetase, hydrolase, manuscript = [tmp_path / name for name in ("synthetase", "hydrolase", "manuscript")]
    for path in (synthetase, hydrolase, manuscript / "figure"):
        path.mkdir(parents=True)
    (manuscript / "main.tex").write_text(r"\includegraphics{figure/s.png}\includegraphics{figure/h.png}")
    png = b"\x89PNG\r\n\x1a\nfixture"
    (synthetase / "s.png").write_bytes(png)
    (hydrolase / "h.png").write_bytes(png)
    result = figure_manifest([synthetase, hydrolase], manuscript)
    assert {row["source"] for row in result["figures"]} == {str(synthetase / "s.png"), str(hydrolase / "h.png")}
    assert not result["all_copies_match"]
    (hydrolase / "s.png").write_bytes(png)
    with pytest.raises(ValueError, match="found 2"):
        figure_manifest([synthetase, hydrolase], manuscript)


def test_frozen_source_hash_resolves_explicit_relocation(tmp_path, monkeypatch):
    monkeypatch.setattr(integration, "ROOT", tmp_path)
    source = tmp_path / "exp_data/study/code/extractor.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"unchanged extractor")
    (tmp_path / "exp_data/REFACTOR_PATHS_20261009.json").write_text(json.dumps({
        "moves": [{"old": "old/code", "new": "exp_data/study/code", "kind": "directory"}]
    }))
    assert integration.digest(tmp_path / "old/code/extractor.py") == hashlib.sha256(source.read_bytes()).hexdigest()
    assert integration.relocated_source(tmp_path / "unmapped.py") == tmp_path / "unmapped.py"
