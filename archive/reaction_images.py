"""Generate 2D native reaction images for evaluated candidates."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("enzyme_agent")


def generate_native_reaction_images(run_dir: Path, all_evals: list[dict], log_fn=None) -> None:
    """Generate 2D native reaction images using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError:
        if log_fn:
            log_fn("  RDKit not available -- skipping native reaction images")
        return

    reactions_dir = run_dir / "reactions"
    reactions_dir.mkdir(exist_ok=True)
    n_images = 0

    for ev in all_evals:
        d = ev.get("data", ev)
        acc = d.get("accession", "")
        if not acc:
            continue

        native_mols: list[tuple[str, "Chem.Mol"]] = []

        native_subs = d.get("native_substrates", [])
        if native_subs:
            for ns in native_subs:
                smi = ns.get("smiles")
                if not smi:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    label = ns.get("name") or smi
                    native_mols.append((label, mol))
        else:
            sub_sim = d.get("substrate_similarity") or {}
            seen = set()
            for pc in sub_sim.get("property_comparisons", []):
                smi = pc.get("native_smiles")
                if smi and smi not in seen:
                    seen.add(smi)
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        native_mols.append((smi, mol))

        if not native_mols:
            continue

        mols = [m for _, m in native_mols]
        legends = [lbl for lbl, _ in native_mols]
        legends = [
            (l[:50] + "...") if len(l) > 50 else l for l in legends
        ]

        try:
            n_cols = min(len(mols), 3)
            img = Draw.MolsToGridImage(
                mols, molsPerRow=n_cols,
                subImgSize=(400, 300), legends=legends,
            )
            img_path = reactions_dir / f"{acc}_native.png"
            img.save(str(img_path))
            d["reaction_image"] = f"reactions/{acc}_native.png"
            n_images += 1
        except Exception as exc:
            logger.debug("Failed to draw native molecules for %s: %s", acc, exc)

    if n_images and log_fn:
        log_fn(f"  Native reaction images: {n_images} generated")
