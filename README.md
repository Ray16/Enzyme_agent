# Enzyme agent workflow

This repository contains the discovery, structural-modeling and design agents,
their shared evidence catalog, validation contracts and tests. The maintained
branch is `main`.

Start a code review with [the review guide](docs/REVIEW_GUIDE.md). It identifies
the active entry points, scientific evidence boundaries, current limitations and
the source-file manifest.

| Component | Purpose | Documentation |
| --- | --- | --- |
| `enzyme_discovery_agent/` | Retrieve evidence and sequences; record catalytic hypotheses and candidate priorities | [Discovery README](enzyme_discovery_agent/README.md) |
| `structural_modeling_agent/` | Decide molecular setups, validate inputs, execute and inspect structural predictions | [Modeling README](structural_modeling_agent/README.md) |
| `enzyme_design_agent/` | Propose and execute protein-design workflows through external backends | [Design README](enzyme_design_agent/README.md) |
| `campaign_catalog/` | Index existing predictions and retain source and setup identity for reuse | [Catalog CLI](campaign_catalog/cli.py) |

## Install and inspect

Python 3.10 or newer is required. A checkout can be installed with:

```bash
python -m pip install -e '.[test,structural-modeling]'
python -m enzyme_discovery_agent --help
python -m structural_modeling_agent --help
python -m enzyme_design_agent --help
```

`pyproject.toml` declares dependencies and optional extras; `uv.lock` records the
resolved versions. With uv, use `uv sync --extra test --extra structural-modeling`.
Optional embedding and design engines require their own environments and model
assets. See the component READMEs before submitting computation.

Copy `.env.example` to a local `.env` and configure the provider you intend to use
before making model calls. Credentials are not part of the repository.

## Tests

```bash
python -m pytest -q
```

The test configuration covers all four components. Provider and execution tests
use test doubles; passing tests does not establish live-service availability or
experimental enzyme activity.

## Discovery and interpretation

```bash
python -m enzyme_discovery_agent \
  --token-budget 100000 --runtime-budget-seconds 1200 \
  "Describe the target transformation and its reactants here"
```

Retrieved records, source-supported catalytic hypotheses, structural measurements
and experimental activity are distinct evidence types. A family match or
favorable predicted pose does not establish target activity. The primary
candidate interface records every reacting partner's proposed role, its source
support, remaining uncertainties and a discriminating structural question.

Large run histories, prediction ensembles, MSAs, model weights and local caches
remain external artifacts. The review guide explains what was included for
review and how to interpret the compact campaign checkpoint.
