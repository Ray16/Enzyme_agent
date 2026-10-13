# Tag stripping report

Source: `panel/panel_as_published.fasta`  ->  `/tmp/claude-21574/-nfs-lambda-stor-01-homes-rzhu-Enzyme-agent/486d3b43-7cd3-41f0-a0cf-a6bacd124ff9/scratchpad/panel_mature_new.fasta`

| name | published | mature | removed | crystal check |
|---|---|---|---|---|
| UMG-SP1 | 438 | 438 | nothing removed | **PASS** — 8S7Z: exact substring, crystal has 0 tag residue(s) before and 8 after |
| UMG-SP2 | 439 | 439 | nothing removed | **PASS** — 8WDW: exact substring, crystal has 23 tag residue(s) before and 0 after |
| UMG-SP3 | 439 | 439 | nothing removed | **PASS** — 9FVF: exact substring, crystal has 0 tag residue(s) before and 0 after |
| SaAmd_candidate_unverified | 436 | 436 | nothing removed | **NO REF** — no crystal reference supplied |
| Aes72 | 297 | 297 | nothing removed | **PASS** — 9IZT: exact substring, crystal has 0 tag residue(s) before and 0 after |
| CpUrethanase | 551 | 551 | nothing removed | **PASS** — 8WV2: exact substring, crystal has 0 tag residue(s) before and 0 after |
| NylA | 493 | 493 | nothing removed | **PASS** — 3A2P: exact substring, crystal has 0 tag residue(s) before and 0 after |
| NylB | 392 | 392 | nothing removed | **NO REF** — no crystal reference supplied |
| NylB_prime | 392 | 392 | nothing removed | **UNVALIDATED** — 1WYB: closest crystal is only 98.2% identical, so a mismatch is point substitutions, not tags — cannot validate tags here |
| NylBp2_Hyb24_D1_mutant | 392 | 392 | nothing removed | **PASS** — 2E8I: exact substring, crystal has 0 tag residue(s) before and 0 after |
| NylC | 355 | 355 | nothing removed | **PASS** — 5XYG: exact substring, crystal has 0 tag residue(s) before and 0 after |
| NylC-TS_D36A_D122G_H130Y_E263Q | 355 | 355 | nothing removed | **PASS** — 5Y0M: exact substring, crystal has 0 tag residue(s) before and 0 after |
| TvgC | 326 | 326 | nothing removed | **NO REF** — no crystal reference supplied |
| CCPUR1 | 518 | 475 | N-term TEV site: -24 `MKHHHHHHPMSDYDIPTTENLYFQ`; C-term TEV site: -15 `ENLYFQGLEHHHHHH`; crystal starts 4 residue(s) later; trimmed leftover linker `GAMG` on crystal evidence | **CORRECTED** — 9TW8: crystal starts 4 residue(s) later; trimmed leftover linker `GAMG` on crystal evidence |
| CCPUR2 | 494 | 475 | C-term TEV site: -15 `ENLYFQGLEHHHHHH`; N-term start inferred from CCPUR1 (89% identical, crystal-anchored): trimmed `MAMG` | **INFERRED** — N-term taken from homolog CCPUR1, whose start is crystal-confirmed. NOT independently validated. |
| MPPUR3 | 519 | 504 | C-term TEV site: -15 `ENLYFQGLEHHHHHH` | **NO REF** — no crystal reference supplied |
| BLPUR4 | 514 | 492 | C-term TEV site (x2, cut before the first): -22 `ENLYFQGENLYFQGLEHHHHHH` | **NO REF** — no crystal reference supplied |
| AXPUR5 | 532 | 517 | C-term TEV site: -15 `ENLYFQGHHHHHHHH` | **NO REF** — no crystal reference supplied |
| AMPUR6 | 494 | 479 | C-term TEV site: -15 `ENLYFQGLEHHHHHH` | **NO REF** — no crystal reference supplied |
| CCPUR7 | 510 | 495 | C-term TEV site: -15 `ENLYFQGLEHHHHHH` | **NO REF** — no crystal reference supplied |
| PMGPUR8 | 534 | 519 | C-term TEV site: -15 `ENLYFQGLEHHHHHH` | **NO REF** — no crystal reference supplied |
| PMGPUR9 | 514 | 499 | C-term TEV site: -15 `ENLYFQGLEHHHHHH` | **NO REF** — no crystal reference supplied |
| PYNYL1 | 498 | 485 | C-term TEV site: -13 `ENLYFQGHHHHHH` | **NO REF** — no crystal reference supplied |
| pQR3139 | 482 | 474 | C-term His tag: -8 `LEHHHHHH` | **NO REF** — no crystal reference supplied |
| pQR3141 | 497 | 489 | C-term His tag: -8 `LEHHHHHH` | **NO REF** — no crystal reference supplied |
| pQR3144 | 479 | 471 | C-term His tag: -8 `LEHHHHHH` | **NO REF** — no crystal reference supplied |
