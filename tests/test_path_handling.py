import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import accusnv_snakemake


class ResolveFromCwdTests(unittest.TestCase):
    def test_absolute_path_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            absolute_path = os.path.join(temp_dir, "output", "file.yaml")
            self.assertEqual(
                accusnv_snakemake.resolve_from_cwd(absolute_path),
                absolute_path,
            )

    def test_relative_path_is_resolved_from_current_working_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                self.assertEqual(
                    accusnv_snakemake.resolve_from_cwd("output/file.yaml"),
                    os.path.join(temp_dir, "output", "file.yaml"),
                )
            finally:
                os.chdir(previous_cwd)

    def test_generated_profile_uses_valid_absolute_configfile_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "scripts").mkdir()
            (root / "config.yaml").write_text(
                "snakefile: Snakefile\nconfigfile: ./experiment_info.yaml\n"
            )
            (root / "experiment_info.yaml").write_text("outdir: test\n")

            previous_cwd = os.getcwd()
            previous_script_dir = accusnv_snakemake.script_dir
            try:
                os.chdir(root)
                accusnv_snakemake.script_dir = str(root)
                for output_dir in ("relative-output", str(root / "absolute-output")):
                    with self.subTest(output_dir=output_dir):
                        temp_output = os.path.join(output_dir, "temp")
                        os.makedirs(temp_output, exist_ok=True)
                        accusnv_snakemake.copy_config_files(
                            "unused.csv", "testuid", output_dir, [], [], 0, temp_output
                        )
                        expected_configfile = os.path.abspath(
                            os.path.join(temp_output, "experiment_info_testuid_tem.yaml")
                        )
                        profile = Path(output_dir) / "conf" / "config.yaml"
                        self.assertIn(
                            f"configfile: {expected_configfile}\n",
                            profile.read_text(),
                        )
                        self.assertNotIn(".//", profile.read_text())
            finally:
                accusnv_snakemake.script_dir = previous_script_dir
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
