#!/usr/bin/env python3
"""
Build GEM Tool
==============

A unified pipeline for building GEMs from genome sequences.

Pipeline:
1. Run GeneMarkS → get protein FASTA
2. Clean FASTA headers (for CarveMe compatibility)
3. Run CarveMe → get draft GEM (SBML .xml)

Functions are modular so they can be called independently,
or chained together for a complete workflow.
"""

import os
import subprocess
from typing import Dict, Optional
from datetime import datetime
from Bio import SeqIO

from .utils.GeneMarkS import GeneMarkSRunner


# ======================
# 1. FASTA Cleaner
# ======================
def clean_faa(input_fasta: str, output_fasta: Optional[str] = None, prefix: str = "protein") -> str:
    """
    Clean headers in a FASTA file and write to a new file.

    Args:
        input_fasta: Path to input protein FASTA (from GeneMarkS).
        output_fasta: Path to save cleaned FASTA (default: add "_clean.fasta").
        prefix: Prefix for gene IDs.

    Returns:
        Path to cleaned FASTA file.
    """
    input_fasta = os.path.abspath(input_fasta)
    if not os.path.exists(input_fasta):
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")

    if output_fasta is None:
        base, ext = os.path.splitext(input_fasta)
        output_fasta = base + "_clean.fasta"

    with open(output_fasta, "w") as out:
        for i, record in enumerate(SeqIO.parse(input_fasta, "fasta"), start=1):
            record.id = f"{prefix}_{i}"
            record.description = ""
            SeqIO.write(record, out, "fasta")

    return output_fasta


# ======================
# 2. CarveMe Runner
# ======================
def run_carveme(input_fasta: str, output_xml: str,
                tmpdir: str = "src/GEMFactory/data/temp",
                gapfill: Optional[str] = None) -> str:
    """
    Run CarveMe to build a GEM.

    Args:
        input_fasta: Cleaned protein FASTA.
        output_xml: Output GEM SBML file path.
        tmpdir: Temporary directory for DIAMOND.
        gapfill: Medium for gap-filling (e.g., "M9,LB"). None = no gap-filling.

    Returns:
        Path to GEM XML file.
    """
    os.makedirs(os.path.dirname(output_xml), exist_ok=True)
    os.makedirs(tmpdir, exist_ok=True)

    command = [
        "carve", input_fasta,
        "-o", output_xml,
        "--diamond-args", f"--tmpdir {tmpdir}"
    ]
    if gapfill:
        command.extend(["-g", gapfill])

    print(f"[{datetime.now()}] Running CarveMe: {' '.join(command)}")

    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"CarveMe failed:\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
        )

    return output_xml


# ======================
# 3. Full Pipeline
# ======================
def build_gem(genome_fasta: str,
              output_dir: str = "src/GEMFactory/data/CarveMe",
              gms_script: Optional[str] = None,
              genome_type: str = "bacteria",
              gcode: str = "11",
              gapfill: Optional[str] = None) -> Dict[str, str]:
    """
    Full pipeline: Genome → GeneMarkS → Clean FASTA → CarveMe GEM

    Args:
        genome_fasta: Input genome file (.fna).
        output_dir: Directory to save GEM.
        gms_script: Path to gms2.pl.
        genome_type: Genome type for GeneMarkS.
        gcode: Genetic code table number.
        gapfill: Medium for gap-filling (e.g., "M9").

    Returns:
        Dict with paths {gff, fnn, faa, clean_faa, gem}
    """
    # 1. Run GeneMarkS
    gms_runner = GeneMarkSRunner(gms_script_path=gms_script)
    gms_outputs = gms_runner.run(
        input_fasta=genome_fasta,
        output_dir="src/GEMFactory/data/GeneMarkS",
        genome_type=genome_type,
        gcode=gcode
    )

    # 2. Clean FASTA
    clean_faa_path = clean_faa(gms_outputs["faa"])

    # 3. Run CarveMe
    prefix = os.path.splitext(os.path.basename(genome_fasta))[0]
    gem_output = os.path.join(output_dir, f"{prefix}_draft.xml")
    run_carveme(clean_faa_path, gem_output, gapfill=gapfill)

    return {
        **gms_outputs,
        "clean_faa": clean_faa_path,
        "gem": gem_output
    }


if __name__ == "__main__":
    # CLI demo
    genome = "src/GEMFactory/data/Genome/GCF_000005845.2_ASM584v2_genomic.fna"
    results = build_gem(genome, gms_script="/home/shenmaa/gms2_linux_64/gms2.pl", gapfill="M9")
    print("\n=== Build GEM Pipeline Finished ===")
    for k, v in results.items():
        print(f"{k}: {v}")
