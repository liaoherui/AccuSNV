import ast
import glob
import pickle
import tempfile
import unittest
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "snv_module_recoded_with_dNdS.py"


class FakeDataFrame:
    def __init__(self, rows=None):
        self.rows = [] if rows is None else rows
        self.shape = (len(self.rows), 0)


class FakePandas:
    @staticmethod
    def Series(dtype=None):
        return object()

    DataFrame = FakeDataFrame


class FakeExaminer:
    @staticmethod
    def available_limits(gff_handle):
        return {"gff_type": {}}


class FakeGFF:
    GFFExaminer = FakeExaminer

    @staticmethod
    def parse(gff_handle, limit_info=None):
        return iter(())


def load_parse_gff():
    source = MODULE_PATH.read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        module = ast.parse(source)
    function = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_gff"
    )
    isolated_module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace = {
        "glob": glob,
        "pickle": pickle,
        "pd": FakePandas(),
        "GFF": FakeGFF,
    }
    exec(compile(isolated_module, str(MODULE_PATH), "exec"), namespace)
    return namespace["parse_gff"]


class AnnotationAlignmentTests(unittest.TestCase):
    def test_contigs_without_gff_records_receive_empty_placeholders(self):
        parse_gff = load_parse_gff()
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_dir = Path(temp_dir)
            (reference_dir / "annotations.gff").write_text("##gff-version 3\n")

            annotations = parse_gff(
                str(reference_dir),
                ["annotated_contig", "unannotated_contig"],
            )

            self.assertEqual(len(annotations), 2)
            self.assertEqual([annotation.shape for annotation in annotations], [(0, 0), (0, 0)])


if __name__ == "__main__":
    unittest.main()
