#!/usr/bin/env python3
"""
FASTA Cleaner Tool for CarveMe
==============================

This script standardizes protein FASTA files (e.g. GeneMarkS output)
so that headers are clean and compatible with CarveMe.

- Input: protein FASTA file (with complex or numeric headers)
- Output: cleaned FASTA file with headers formatted as ">gene_1", ">gene_2", ...

Usage:
    python clean_faa.py -i input.faa -o output_clean.faa
"""

import argparse
from Bio import SeqIO
import os

def clean_faa(input_fasta: str, output_fasta: str, prefix: str = "protein"):
    """Clean headers in a FASTA file and write to a new file."""
    dir_path = os.path.join("src/GEMFactory/data/GeneMarkS/", input_fasta.split("/")[-1].split("_protein.fasta")[0])
    input_fasta = os.path.join(dir_path, input_fasta)
    with open(os.path.join(dir_path, output_fasta), "w") as out:
        for i, record in enumerate(SeqIO.parse(input_fasta, "fasta"), start=1):
            record.id = f"{prefix}_{i}"   # e.g. protein_1
            record.description = ""       # remove original description
            SeqIO.write(record, out, "fasta")

def main():
    parser = argparse.ArgumentParser(
        description="Clean FASTA headers for CarveMe compatibility"
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input protein FASTA file")
    parser.add_argument("-o", "--output", required=True,
                        help="Output cleaned FASTA file")
    parser.add_argument("--prefix", default="protein",
                        help="Prefix for protein IDs (default: protein)")
    
    args = parser.parse_args()
    clean_faa(args.input, args.output, args.prefix)
    print(f"Cleaned FASTA saved to: {args.output}")

if __name__ == "__main__":
    main()
