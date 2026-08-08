#!/bin/bash
# Two-pass OPC solvation of NylC holo (protein chains A+B + nylon-6,6 ligand LIG, +2).
# Pass 1: solvate, count waters. Pass 2: neutralize + 0.15 M NaCl scaled to water count.
set -e
export AMBERHOME=~/miniforge3
TLEAP=~/miniforge3/bin/tleap
cd /nfs/lambda_stor_01/homes/rzhu/Biochemical_agent/md/NylC_build
OUT=/nfs/lambda_stor_01/homes/rzhu/Biochemical_agent/md

HEAD='source leaprc.protein.ff19SB
source leaprc.water.opc
source leaprc.gaff2
loadamberparams frcmod.ionslm_126_opc
loadamberparams lig.frcmod
LIG = loadmol2 lig.mol2
PROT = loadpdb prot_clean.pdb
COMPLEX = combine { PROT LIG }
check COMPLEX
solvatebox COMPLEX OPCBOX 12.0'

# ---- Pass 1: solvate only, count waters ----
cat > tleap_pass1.in <<EOF
$HEAD
savepdb COMPLEX pass1.pdb
quit
EOF
$TLEAP -f tleap_pass1.in > tleap_pass1.log 2>&1
NWAT=$(grep -c ' O   WAT\| EPW WAT\|^ATOM.* O   WAT' pass1.pdb || true)
# robust water count: count OPC oxygens (atom name O, res WAT)
NWAT=$(awk '$1=="ATOM"||$1=="HETATM"{ if($4=="WAT" && $3=="O") c++ } END{print c+0}' pass1.pdb)
echo "Pass1 waters (OPC molecules): $NWAT"
# 0.15 M NaCl: n = 0.15/55.5 * nwat
NION=$(python3 -c "print(round(0.15/55.5*$NWAT))")
echo "0.15 M NaCl ion count each: $NION"

# ---- Pass 2: neutralize + salt ----
cat > tleap_pass2.in <<EOF
$HEAD
addions COMPLEX Na+ 0
addions COMPLEX Cl- 0
addIonsRand COMPLEX Na+ $NION Cl- $NION
savepdb COMPLEX $OUT/NylC_holo_solvated.pdb
saveamberparm COMPLEX $OUT/NylC_holo.prmtop $OUT/NylC_holo.inpcrd
quit
EOF
$TLEAP -f tleap_pass2.in > tleap_pass2.log 2>&1
echo "=== FATAL check ==="; grep FATAL tleap_pass2.log || echo "no FATAL"
echo "=== tleap final summary ==="; grep -iE 'Added|charge|atoms|residues' tleap_pass2.log | tail -20
