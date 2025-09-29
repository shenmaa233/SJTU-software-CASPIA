import os
import subprocess
import argparse
from typing import Optional, Dict
from datetime import datetime


class GeneMarkSRunner:
    """
    GeneMarkS Linux runner tool.
    Used for genome annotation, outputs gff/fnn/faa files for downstream tools such as CarveMe.
    """

    def __init__(self, gms_script_path: Optional[str] = None, log_dir: str = "./logs"):
        """
        Initialize the GeneMarkS runner.
        :param gms_script_path: Path to the GeneMarkS script (gms2.pl). If None, use the default path.
        :param log_dir: Directory for log output.
        """
        self.gms_script_path = gms_script_path
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        if not self.gms_script_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.gms_script_path = os.path.join(current_dir, "gms2_linux_64", "gms2.pl")

    def log(self, message: str):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        with open(os.path.join(self.log_dir, "genemarks.log"), "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def run(self, input_fasta: str, output_dir: str,
            genome_type: str = "bacteria", gcode: str = "11") -> Dict[str, str]:
        """
        Run the GeneMarkS annotation process.
        :param input_fasta: Input genome fasta file.
        :param output_dir: Output directory.
        :param genome_type: Genome type (bacteria, archaea, etc.).
        :param gcode: Genetic code table (commonly 11 for bacteria).
        :return: Dictionary of output file paths {gff, fnn, faa}.
        """
        output_dir = os.path.join(output_dir, input_fasta.split("/")[-1].split(".fna")[0])
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(input_fasta):
            raise FileNotFoundError(f"Input file does not exist: {input_fasta}")

        if not os.path.exists(self.gms_script_path):
            raise FileNotFoundError(f"GeneMarkS script does not exist: {self.gms_script_path}")

        input_fasta = os.path.abspath(input_fasta)
        output_dir = os.path.abspath(output_dir)

        prefix = os.path.splitext(os.path.basename(input_fasta))[0]
        output_files = {
            "gff": os.path.join(output_dir, f"{prefix}.gff"),
            "fnn": os.path.join(output_dir, f"{prefix}_gene.fasta"),
            "faa": os.path.join(output_dir, f"{prefix}_protein.fasta"),
        }

        command = [
            "perl", self.gms_script_path,
            "--seq", input_fasta,
            "--genome-type", genome_type,
            "--gcode", gcode,
            "--format", "gff",
            "--output", output_files["gff"],
            "--fnn", output_files["fnn"],
            "--faa", output_files["faa"],
        ]

        self.log(f"Executing command: {' '.join(command)}")

        try:
            temp_dir = os.path.abspath("src/GEMFactory/data/temp")
            os.makedirs(temp_dir, exist_ok=True)

            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=temp_dir
            )
            self.log("stdout:\n" + process.stdout)
            self.log("stderr:\n" + process.stderr)
            if process.returncode == 0:
                self.log("GeneMarkS annotation completed successfully")
                return output_files
            else:
                raise RuntimeError(f"GeneMarkS execution failed, return code: {process.returncode}")
        except Exception as e:
            raise RuntimeError(f"Error occurred while running GeneMarkS: {e}") from e


def main():
    parser = argparse.ArgumentParser(
        description="GeneMarkS Linux genome annotation tool (can be used with CarveMe downstream)"
    )
    parser.add_argument("-i", "--input", required=True, help="Input genome fasta file path")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--script", default=None, help="GeneMarkS script path (auto-detect by default)")
    parser.add_argument("--genome-type", default="bacteria", help="Genome type (default: bacteria)")
    parser.add_argument("--gcode", default="11", help="Genetic code table number (default: 11)")

    args = parser.parse_args()

    runner = GeneMarkSRunner(gms_script_path=args.script)
    results = runner.run(
        input_fasta=args.input,
        output_dir=args.output,
        genome_type=args.genome_type,
        gcode=args.gcode
    )

    print("\n=== Annotation completed, output files ===")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
