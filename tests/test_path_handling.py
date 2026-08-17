"""The generated config must not depend on where the user typed the command.

Workflow jobs run with --directory set to the output directory, so every path the config hands
to Snakemake has to be absolute regardless of whether -o was relative.
"""
import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from accusnv import cli


def create_configs(output_dir, mode="local", **overrides):
    """Write the runtime configs for a run into <output_dir>/configs and return them parsed."""
    settings = dict(output_dir=output_dir, mode=mode, config_file=None, pipeline_file=None,
                    cores=None, partition=None, env=None,
                    exclude_positions=None, include_positions=None,
                    **{flag: False for flag in cli.FLAG_KEYS})
    args = argparse.Namespace(**{**settings, **overrides})
    conf_dir = os.path.join(output_dir, "configs")
    os.makedirs(conf_dir, exist_ok=True)
    pipeline_file, config_file = cli.create_configs(args, argparse.ArgumentParser(), conf_dir)
    with open(config_file) as config, open(pipeline_file) as pipeline:
        return yaml.safe_load(config), yaml.safe_load(pipeline)


class ConfigPathTests(unittest.TestCase):
    def test_run_paths_are_absolute_for_relative_and_absolute_output_dirs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                for output_dir in ("relative-output", os.path.join(temp_dir, "absolute-output")):
                    with self.subTest(output_dir=output_dir):
                        config, _ = create_configs(output_dir)
                        expected = os.path.abspath(output_dir)

                        self.assertEqual(config["configfile"],
                                         os.path.join(expected, "configs", "pipeline.yaml"))
                        self.assertEqual(sorted(config["config"]),
                                         [f"env=", f"outdir={expected}",
                                          f"sample_table={os.path.join(expected, 'samples.csv')}"])
                        for value in [config["configfile"]] + config["config"]:
                            self.assertNotIn(".//", value)
            finally:
                os.chdir(previous_cwd)

    def test_local_mode_drops_the_slurm_settings_the_template_ships(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config, pipeline = create_configs(temp_dir, cores=8)

            self.assertEqual(config["executor"], "local")
            self.assertEqual(config["cores"], 8)      # matches the threads the mapping rules ask for
            self.assertEqual(pipeline["cores"], 8)
            self.assertNotIn("jobs", config)
            self.assertEqual([r for r in config["default-resources"] if r.startswith("slurm_")], [])

    def test_slurm_mode_keeps_the_executor_and_takes_the_partition(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config, _ = create_configs(temp_dir, mode="slurm", partition="short,long")

            self.assertEqual(config["executor"], "slurm")
            self.assertNotIn("cores", config)         # SLURM runs are bounded by jobs, not cores
            self.assertIn('slurm_partition="short,long"', config["default-resources"])

    def test_position_lists_reach_the_pipeline_config_as_absolute_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                Path("exclude.txt").write_text("1000\n")
                _, pipeline = create_configs("out", exclude_positions="exclude.txt")

                self.assertEqual(pipeline["exclude_positions"],
                                 os.path.join(temp_dir, "exclude.txt"))
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
