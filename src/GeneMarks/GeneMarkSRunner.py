import os
import subprocess
import argparse
from typing import Optional, Dict
from datetime import datetime


class GeneMarkSRunner:
    """
    GeneMarkS Linux 运行工具
    用于基因组注释，输出 gff/fnn/faa 文件，供 CarveMe 等下游工具使用。
    """

    def __init__(self, gms_script_path: Optional[str] = None, log_dir: str = "./logs"):
        """
        初始化 GeneMarkS 运行器
        :param gms_script_path: GeneMarkS 脚本路径 (gms2.pl)，如果为空则使用默认路径
        :param log_dir: 日志输出目录
        """
        self.gms_script_path = gms_script_path
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        if not self.gms_script_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.gms_script_path = os.path.join(current_dir, "gms2_linux_64", "gms2.pl")

    def log(self, message: str):
        """日志输出到屏幕和文件"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        with open(os.path.join(self.log_dir, "genemarks.log"), "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def run(self, input_fasta: str, output_dir: str,
            genome_type: str = "bacteria", gcode: str = "11") -> Dict[str, str]:
        """
        运行 GeneMarkS 注释流程
        :param input_fasta: 输入基因组 fasta 文件
        :param output_dir: 输出目录
        :param genome_type: 基因组类型 (bacteria, archaea 等)
        :param gcode: 遗传密码表 (bacteria 常用 11)
        :return: 输出文件路径字典 {gff, fnn, faa}
        """
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(input_fasta):
            raise FileNotFoundError(f"输入文件不存在: {input_fasta}")

        if not os.path.exists(self.gms_script_path):
            raise FileNotFoundError(f"GeneMarkS脚本不存在: {self.gms_script_path}")

        # 输出文件名
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
            "--faa", output_files["faa"]
        ]

        self.log(f"执行命令: {' '.join(command)}")

        try:
            process = subprocess.run(command, capture_output=True, text=True)
            self.log("stdout:\n" + process.stdout)
            self.log("stderr:\n" + process.stderr)
            if process.returncode == 0:
                self.log("GeneMarkS 注释成功完成")
                return output_files
            else:
                raise RuntimeError(f"GeneMarkS 执行失败，返回码: {process.returncode}")
        except Exception as e:
            raise RuntimeError(f"运行 GeneMarkS 时出错: {e}") from e


def main():
    parser = argparse.ArgumentParser(
        description="GeneMarkS Linux 基因组注释工具 (下游可对接 CarveMe)"
    )
    parser.add_argument("-i", "--input", required=True, help="输入基因组 fasta 文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("--script", default=None, help="GeneMarkS 脚本路径 (默认自动寻找)")
    parser.add_argument("--genome-type", default="bacteria", help="基因组类型 (默认: bacteria)")
    parser.add_argument("--gcode", default="11", help="遗传密码表编号 (默认: 11)")

    args = parser.parse_args()

    runner = GeneMarkSRunner(gms_script_path=args.script)
    results = runner.run(
        input_fasta=args.input,
        output_dir=args.output,
        genome_type=args.genome_type,
        gcode=args.gcode
    )

    print("\n=== 运行完成，输出文件 ===")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
