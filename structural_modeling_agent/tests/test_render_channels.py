"""Light tests for render_channels helpers (no pymol needed): the tunnel-radius parse (B-factor column),
the CAVER cluster-id -> file mapping, and the shared colormap. The full render is exercised manually."""
import os
import pytest

from structural_modeling_agent.analysis import render_channels as rc


def test_radii_read_from_bfactor_column(tmp_path):
    # CAVER sphere format: radius in the B-factor column (61-66), occupancy blank
    pdb = tmp_path / "tun.pdb"
    pdb.write_text(
        "ATOM      1  H   FIL T   1      24.587  94.960  38.384        1.36\n"
        "ATOM      2  H   FIL T   1      25.347  95.035  37.997        1.80\n"
        "CONECT    1    2\n"
        "ATOM      3  H   FIL T   1      25.363  94.894  38.987        2.31\n")
    assert rc._radii(str(pdb)) == [1.36, 1.80, 2.31]


def test_tunnel_pdb_mapping(tmp_path):
    data = tmp_path / "clusters_timeless"
    data.mkdir()
    (data / "tun_cl_002_1.pdb").write_text("END\n")
    assert rc._tunnel_pdb(str(tmp_path), 2).endswith("tun_cl_002_1.pdb")
    assert rc._tunnel_pdb(str(tmp_path), 7) is None          # absent -> None, not a guess


def test_sphere_colors_span_the_ramp():
    cols = rc._sphere_colors([rc.RADIUS_VMIN, rc.RADIUS_VMAX])
    assert len(cols) == 2 and all(len(c) == 3 for c in cols)
    assert cols[0] != cols[1]                                 # narrow and wide map to different colours


def test_fast_channels_html_viewer_end_to_end(tmp_path):
    pytest.importorskip("MDAnalysis")
    from structural_modeling_agent.analysis import channels_html, substrate_channels as ch
    dhaa = os.path.join(os.path.dirname(__file__), "fixtures", "dhaa_1cqw.pdb")
    if not os.path.exists(dhaa):
        pytest.skip("DhaA fixture not available")
    run_dir = str(tmp_path / "dhaa_fast")
    ch.analyze_channels([dhaa], run_dir, start_sel="resid 117 and name OD1 OD2", n_starts=1, backend="fast")
    out_html = str(tmp_path / "dhaa_channels.html")
    channels_html.build([("DhaA 1CQW", [("fast", run_dir)])], out_html, cdn=True)
    text = open(out_html).read()
    assert "DhaA 1CQW" in text and "spheres" in text and len(text) > 5000
