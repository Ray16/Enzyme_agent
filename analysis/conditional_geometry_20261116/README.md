# Conditional geometry and step-2 distance scatter

An exploratory alternative to the original all-pose angle plot, using the saved November 6 discovery run analysis. The historical plot and frozen score are retained.

- X: inverse-distance-weighted step-1 angle, computed only among poses with step-1 attack distance strictly less than 4 Å. No angle filtering.
- Y: every saved step-2 amine N–carbonyl C distance; the bubble is centered at the median. Smaller distances appear higher.
- Bubble area: proportion of the 50 step-1 poses with attack distance below 4 Å.
- Right strip: candidates with zero close step-1 poses, hence undefined conditional angle. Crosses mark medians and do not encode fractions.
- Horizontal jitter is for visibility only. Color distinguishes candidates; YjfC is highlighted in magenta.

All 46 paired candidates have 50 identity-passing poses per step. Seven have no close step-1 poses. Step 1 uses the unrestrained ensemble; step 2 uses the pocket-guided localization ensemble. These are separate ensembles, not jointly paired observations or joint reaction probabilities. No catalytic activity ranking is inferred. Conditional angles based on very few poses should be interpreted with their small bubble sizes. The 4 Å cutoff is an explicit exploratory convention; threshold sensitivity has not been evaluated in this figure.

The source CSV checksums and exact metric are in provenance.json. conditional_geometry_data.csv contains all displayed summaries, counts, step-2 close fractions and the original frozen angles. The original per-pose CSV remains the source of the distance scatter.

Reproduce from repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python analysis/conditional_geometry_20261116/plot_conditional_geometry.py
```

Verification: all ensemble sizes and identity flags checked; YjfC conditional angle independently matched to 167.77612327325429 degrees and its close-pose count to 16/50; PNG visually inspected for labels, size legend and undefined-angle handling.

## Step-1 cutoff sensitivity: 6 Å

The `step1_6A/` outputs change only the step-1 conditioning threshold and bubble-area fraction to distance <6 Å. All step-2 distances, medians, reported <4 Å fractions, and the horizontal 4 Å reference remain unchanged. AS9 is marked PA55-active based on the user’s experimental identification. Forty-five candidates have defined conditional angles; GlnA2 Q9RDS6 remains in the undefined strip.

AS9: 18/50 poses (36%), conditional angle 166.1545°, median step-2 distance 9.8989 Å. YjfC: 50/50 poses (100%), conditional angle 138.0290°, median step-2 distance 12.3284 Å. This threshold sensitivity is descriptive and does not validate a new activity ranking.

Reproduce with `--step1-cutoff 6`. The original 4 Å outputs are preserved. The revised script labels the threshold explicitly and keeps candidate colors stable across newly generated cutoff variants.
