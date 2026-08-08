"""CLI entry point for the enzyme discovery agent.

Usage:
    python -m enzyme_discovery_agent "find hydrolases for a target bond-cleavage reaction"
    python -m enzyme_discovery_agent --substrate "CC(=O)N" "find enzymes for acetamide hydrolysis"
    python -m enzyme_discovery_agent --structure-file substrate.mol "find enzymes for this substrate"
"""

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

from enzyme_discovery_agent import config
from enzyme_discovery_agent.agent import EnzymeAgent
from enzyme_discovery_agent.models.target_profile import load_target_profile

_SKETCH_URL = "https://pubchem.ncbi.nlm.nih.gov/edit3/index.html"


def _load_structure_file(path: str) -> tuple[str, str | None]:
    from enzyme_discovery_agent.molecule_input import read_single_molecule
    rec = read_single_molecule(path)
    return rec["smiles"], rec["name"]


def _launch_and_load(url: str, tool_name: str) -> tuple[str, str | None]:
    print(f"Opening {tool_name} in your browser...")
    print(f"  URL: {url}")
    webbrowser.open(url)
    print()
    print("Draw your molecule, then save/download the structure file")
    print("(MOL, SDF, MOL2, PDB, SMI, or InChI format).")
    print()
    path = input("Enter path to the downloaded structure file: ").strip()
    if not path:
        print("No file path provided. Exiting.", file=sys.stderr)
        sys.exit(1)
    return _load_structure_file(path)


def main():
    parser = argparse.ArgumentParser(
        description="Enzyme Discovery Agent -- LLM-driven enzyme search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python -m enzyme_discovery_agent "find hydrolases for a target bond-cleavage reaction"\n'
            '  python -m enzyme_discovery_agent --substrate "CC(=O)N" "find enzymes for acetamide hydrolysis"\n'
            '  python -m enzyme_discovery_agent --structure-file substrate.mol "find enzymes"\n'
        ),
    )
    parser.add_argument("query", help="Natural language query describing what enzymes to find")

    substrate_group = parser.add_mutually_exclusive_group()
    substrate_group.add_argument("--substrate", type=str, default=None, help="Substrate SMILES string")
    substrate_group.add_argument("--structure-file", type=str, default=None,
                                help="Path to a molecule structure file (.mol, .sdf, .mol2, .pdb, .smi, .inchi)")
    substrate_group.add_argument("--sketch", action="store_true", default=False,
                                help="Open PubChem Sketch to draw a molecule")

    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL, help="Model to use via Argo")
    parser.add_argument("--output-dir", type=str, default=None, help="Base directory for run outputs")
    parser.add_argument("--max-iterations", type=int, default=config.MAX_ITERATIONS,
                        help=f"Maximum agent iterations (default: {config.MAX_ITERATIONS})")
    parser.add_argument("--llm-timeout", type=int, default=120, help="Timeout in seconds per LLM API call")
    parser.add_argument("--target-profile", type=str, default=None,
                        help=("Path to a JSON target profile with generic evidence requirements and "
                              "optional pre-ranking discovery coverage; see target_profile.example.json"))

    args = parser.parse_args()

    substrate_smiles = args.substrate
    substrate_name = None

    if args.structure_file:
        substrate_smiles, substrate_name = _load_structure_file(args.structure_file)
        print(f"Loaded substrate: {substrate_name or 'unnamed'} ({substrate_smiles})")
    elif args.sketch:
        substrate_smiles, substrate_name = _launch_and_load(_SKETCH_URL, "PubChem Sketch")
        print(f"Loaded substrate: {substrate_name or 'unnamed'} ({substrate_smiles})")

    target_profile = None
    if args.target_profile:
        try:
            target_profile = load_target_profile(args.target_profile)
        except (OSError, ValueError) as exc:
            parser.error(f"Invalid target profile: {exc}")

    try:
        agent = EnzymeAgent(
            model=args.model,
            output_dir=args.output_dir or Path(__file__).parent / "runs",
            max_iterations=args.max_iterations,
            llm_timeout=args.llm_timeout,
            query=args.query,
        )
    except RuntimeError as exc:
        parser.error(str(exc))

    report = agent.run(
        query=args.query,
        substrate_smiles=substrate_smiles,
        substrate_name=substrate_name,
        target_profile=target_profile,
    )

    agent_logger = logging.getLogger("enzyme_agent")
    agent_logger.info("\n" + "=" * 80)
    agent_logger.info("FINAL REPORT")
    agent_logger.info("=" * 80)
    agent_logger.info(report)


if __name__ == "__main__":
    main()
