#!/bin/bash
# Run the ALF TwoBand_Spin model for a series of time steps, with the same physical
# parameters as a SLED run, for scripts/compare_alf.py.
#
# Usage:  scripts/run_alf_series.sh OUT_DIR COUPLING "DTAUS" [L1 L2 BETA CHEM U NBIN NSWEEP]
# e.g.    scripts/run_alf_series.sh runs/alf_heis Heisenberg "0.05 0.025 0.0125" 4 4 4.0 0.3 1.0 25 40
#
# Each time step goes to OUT_DIR/dtau_<dtau>/ (ALF run directory, analysed by ana.out).
# The ALF installation is ../../ALF_SZ/ALF relative to SLED (never $ALF_DIR).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALF="$(cd "$HERE/../../../ALF_SZ/ALF" && pwd)"
TEMPLATE="$ALF/Scripts_and_Parameters_files/TwoBand_Spin/parameters"
RUN="$ALF/Scripts_and_Parameters_files/TwoBand_Spin/run.sh"
OUT=$1; COUP=$2; DTAUS=$3
L1=${4:-4}; L2=${5:-4}; BETA=${6:-4.0}; CHEM=${7:-0.3}; U=${8:-1.0}; NBIN=${9:-25}; NSWEEP=${10:-40}
mkdir -p "$OUT"
for DT in $DTAUS; do
  D="$OUT/dtau_$(printf '%g' "$DT")"
  mkdir -p "$D"
  sed -e "s/^L1 *=.*/L1 = $L1/" -e "s/^L2 *=.*/L2 = $L2/" \
      -e "s/^Ham_Int *=.*/Ham_Int = \"$COUP\"/" -e "s/^Ham_chem *=.*/Ham_chem = ${CHEM}d0/" \
      -e "s/^Ham_U *=.*/Ham_U = ${U}d0/" -e "s/^Dtau *=.*/Dtau = ${DT}d0/" \
      -e "s/^Beta *=.*/Beta = ${BETA}d0/" -e "s/^NSweep *=.*/NSweep = $NSWEEP/" \
      -e "s/^NBin *=.*/NBin = $NBIN/" "$TEMPLATE" > "$D/parameters.in"
  ( "$RUN" "$D" "$D/parameters.in" > "$D/driver.log" 2>&1 && echo "done $D" ) &
done
wait
