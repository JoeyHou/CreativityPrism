"""Phase 2A regressions: artifact markers, centralized outputs, metadata."""
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from runner import artifacts
from runner import run as runner_module

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = REPO_ROOT / "registry" / "adapters"

INVOCATION = {
    "inference_model": "GPT4.1",
    "judge_model": "GPT4.1-mini",
    "limit": 5,
    "mode": "both",
}
TASK_META = {
    "name": "aut",
    "environment": "modern",
    "adapter": "registry/adapters/aut.sh",
}


class MarkerParsingTests(unittest.TestCase):
    def test_both_kinds_are_parsed(self):
        artifacts_map, warnings = artifacts.parse_artifact_markers([
            "loading model...\n",
            "CP_ARTIFACT inference /native/inference\n",
            "CP_ARTIFACT eval /native/eval.json\n",
            "done\n",
        ])
        self.assertEqual(
            artifacts_map,
            {"inference": "/native/inference", "eval": "/native/eval.json"},
        )
        self.assertEqual(warnings, [])

    def test_legacy_output_path_maps_to_inference(self):
        artifacts_map, warnings = artifacts.parse_artifact_markers(
            ["OUTPUT_PATH=/native/legacy\n"]
        )
        self.assertEqual(artifacts_map, {"inference": "/native/legacy"})
        self.assertEqual(warnings, [])

    def test_explicit_marker_wins_over_legacy_regardless_of_order(self):
        for lines in (
            ["OUTPUT_PATH=/legacy\n", "CP_ARTIFACT inference /new\n"],
            ["CP_ARTIFACT inference /new\n", "OUTPUT_PATH=/legacy\n"],
        ):
            with self.subTest(lines=lines):
                artifacts_map, _ = artifacts.parse_artifact_markers(lines)
                self.assertEqual(artifacts_map, {"inference": "/new"})

    def test_last_marker_of_a_kind_wins(self):
        artifacts_map, _ = artifacts.parse_artifact_markers([
            "CP_ARTIFACT inference /first\n",
            "CP_ARTIFACT inference /second\n",
        ])
        self.assertEqual(artifacts_map, {"inference": "/second"})

    def test_paths_with_spaces_are_preserved(self):
        artifacts_map, _ = artifacts.parse_artifact_markers(
            ["CP_ARTIFACT inference /native/my outputs/run.json\n"]
        )
        self.assertEqual(artifacts_map, {"inference": "/native/my outputs/run.json"})

    def test_unknown_kind_and_missing_path_warn_without_registering(self):
        artifacts_map, warnings = artifacts.parse_artifact_markers([
            "CP_ARTIFACT scores /native/scores.json\n",
            "CP_ARTIFACT inference\n",
        ])
        self.assertEqual(artifacts_map, {})
        self.assertEqual(len(warnings), 2)

    def test_marker_like_prose_is_ignored(self):
        artifacts_map, warnings = artifacts.parse_artifact_markers([
            "see OUTPUT_PATH=/not/a/marker for details\n",
            "the adapter should print CP_ARTIFACT inference <path>\n",
        ])
        self.assertEqual(artifacts_map, {})
        self.assertEqual(warnings, [])


class MaterializeTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.outputs = self.root / "outputs"
        self.native = self.root / "native"
        self.native.mkdir()
        self.addCleanup(self._tmp.cleanup)

    def native_file(self, name="inference_output.json", content="[]"):
        path = self.native / name
        path.write_text(content, encoding="utf-8")
        return path

    def native_dir(self, name="aut_run"):
        path = self.native / name
        path.mkdir()
        (path / "part.json").write_text("[]", encoding="utf-8")
        return path

    def materialize(self, lines, label="v1", task_meta=None, invocation=None,
                    exit_code=0):
        return artifacts.materialize_run_outputs(
            self.outputs,
            label,
            task_meta or TASK_META,
            invocation or INVOCATION,
            ["bash", "adapter.sh"],
            exit_code,
            lines,
        )

    def resolve_link(self, target_dir, record):
        """Resolve either representation to the native path it points at."""
        link = target_dir / record["link"]
        if record["link_type"] == "reference":
            link = Path(link.read_text(encoding="utf-8").strip())
        return Path(os.path.realpath(link))


class LinkingTests(MaterializeTestCase):
    def test_file_artifact_is_linked_not_copied(self):
        native = self.native_file()
        target_dir, metadata, warnings = self.materialize(
            [f"CP_ARTIFACT inference {native}\n"]
        )
        record = metadata["artifacts"]["inference"]

        self.assertTrue(record["exists"])
        self.assertEqual(record["native_type"], "file")
        self.assertIn(record["link_type"], ("symlink", "reference"))
        self.assertEqual(
            self.resolve_link(target_dir, record), Path(os.path.realpath(native))
        )
        self.assertEqual(native.read_text(encoding="utf-8"), "[]")
        self.assertNotIn("no artifacts", " ".join(warnings))

    def test_directory_artifact_link_has_no_forced_extension(self):
        native = self.native_dir()
        target_dir, metadata, _ = self.materialize(
            [f"CP_ARTIFACT inference {native}\n"]
        )
        record = metadata["artifacts"]["inference"]

        self.assertEqual(record["native_type"], "directory")
        if record["link_type"] == "symlink":
            self.assertEqual(record["link"], "inference_output")
        self.assertEqual(
            self.resolve_link(target_dir, record), Path(os.path.realpath(native))
        )

    def test_native_outputs_are_never_moved_or_duplicated(self):
        native = self.native_file(content='{"kept": true}')
        target_dir, _, _ = self.materialize([f"CP_ARTIFACT inference {native}\n"])

        self.assertTrue(native.is_file())
        self.assertEqual(native.read_text(encoding="utf-8"), '{"kept": true}')
        for entry in target_dir.iterdir():
            if entry.name == artifacts.METADATA_FILENAME:
                continue
            self.assertTrue(
                entry.is_symlink() or entry.suffix == ".path",
                f"{entry.name} is a real copy, not a link or reference",
            )


class SymlinkTargetTests(MaterializeTestCase):
    def test_in_repo_artifacts_get_a_relative_target(self):
        base = self.root
        native = self.native_file()
        link_dir = artifacts.centralized_output_dir(
            self.outputs, "v1", "aut", "GPT4.1"
        )
        target = artifacts._symlink_target(native, link_dir, base.resolve())

        self.assertFalse(os.path.isabs(target))
        self.assertEqual(
            os.path.realpath(os.path.join(link_dir, target)),
            os.path.realpath(native),
        )

    def test_out_of_repo_artifacts_get_an_absolute_target(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            native = Path(elsewhere) / "run.json"
            native.write_text("[]", encoding="utf-8")
            link_dir = artifacts.centralized_output_dir(
                self.outputs, "v1", "aut", "GPT4.1"
            )
            target = artifacts._symlink_target(native, link_dir, self.root.resolve())

            self.assertTrue(os.path.isabs(target))
            self.assertEqual(target, str(native))


class PathIsolationTests(MaterializeTestCase):
    def test_label_task_and_model_each_get_their_own_directory(self):
        native = self.native_file()
        line = f"CP_ARTIFACT inference {native}\n"
        seen = set()
        for label in ("v1", "v2"):
            for task in ("aut", "ttcw"):
                for model in ("GPT4.1", "Qwen2.5-72B"):
                    target_dir, _, _ = self.materialize(
                        [line],
                        label=label,
                        task_meta={**TASK_META, "name": task},
                        invocation={**INVOCATION, "inference_model": model},
                    )
                    self.assertNotIn(target_dir, seen)
                    seen.add(target_dir)
                    self.assertEqual(
                        target_dir, self.outputs / label / task / model
                    )
        self.assertEqual(len(seen), 8)

    def test_sibling_runs_do_not_share_artifact_records(self):
        first = self.native_file("first.json")
        second = self.native_file("second.json")
        dir_a, meta_a, _ = self.materialize([f"CP_ARTIFACT inference {first}\n"])
        dir_b, meta_b, _ = self.materialize(
            [f"CP_ARTIFACT inference {second}\n"],
            invocation={**INVOCATION, "inference_model": "Qwen2.5-72B"},
        )

        self.assertNotEqual(dir_a, dir_b)
        self.assertEqual(meta_a["artifacts"]["inference"]["native_path"], str(first))
        self.assertEqual(meta_b["artifacts"]["inference"]["native_path"], str(second))


class MetadataTests(MaterializeTestCase):
    def test_metadata_records_the_run_context(self):
        native = self.native_file()
        target_dir, metadata, _ = self.materialize(
            [f"CP_ARTIFACT inference {native}\n"]
        )
        on_disk = json.loads(
            (target_dir / artifacts.METADATA_FILENAME).read_text(encoding="utf-8")
        )

        self.assertEqual(on_disk, metadata)
        self.assertEqual(metadata["schema_version"], artifacts.METADATA_SCHEMA_VERSION)
        self.assertEqual(metadata["label"], "v1")
        self.assertEqual(metadata["task"], "aut")
        self.assertEqual(metadata["inference_model"], "GPT4.1")
        self.assertEqual(metadata["judge_model"], "GPT4.1-mini")
        self.assertEqual(metadata["limit"], 5)
        self.assertEqual(metadata["mode"], "both")
        self.assertEqual(metadata["environment"], "modern")
        self.assertEqual(metadata["adapter"], "registry/adapters/aut.sh")
        self.assertEqual(metadata["command"], ["bash", "adapter.sh"])
        self.assertEqual(metadata["exit_code"], 0)
        self.assertTrue(metadata["created_at"].endswith("Z"))

    def test_failed_adapter_run_is_recorded(self):
        _, metadata, warnings = self.materialize([], exit_code=3)
        self.assertEqual(metadata["exit_code"], 3)
        self.assertEqual(metadata["artifacts"], {})
        self.assertTrue(any("announced no artifacts" in w for w in warnings))

    def test_corrupt_metadata_is_replaced_not_fatal(self):
        native = self.native_file()
        target_dir = artifacts.centralized_output_dir(
            self.outputs, "v1", "aut", "GPT4.1"
        )
        target_dir.mkdir(parents=True)
        (target_dir / artifacts.METADATA_FILENAME).write_text("{not json", encoding="utf-8")

        _, metadata, _ = self.materialize([f"CP_ARTIFACT inference {native}\n"])
        self.assertIn("inference", metadata["artifacts"])


class MissingArtifactTests(MaterializeTestCase):
    def test_no_markers_warns_and_writes_empty_artifacts(self):
        target_dir, metadata, warnings = self.materialize(["nothing to see\n"])

        self.assertEqual(metadata["artifacts"], {})
        self.assertTrue(any("announced no artifacts" in w for w in warnings))
        self.assertTrue((target_dir / artifacts.METADATA_FILENAME).is_file())

    def test_nonexistent_path_is_recorded_without_a_link(self):
        missing = self.native / "not_written.json"
        target_dir, metadata, warnings = self.materialize(
            [f"CP_ARTIFACT inference {missing}\n"]
        )
        record = metadata["artifacts"]["inference"]

        self.assertFalse(record["exists"])
        self.assertIsNone(record["link"])
        self.assertEqual(record["link_type"], "missing")
        self.assertEqual(record["native_path"], str(missing))
        self.assertTrue(any("does not exist" in w for w in warnings))
        self.assertEqual(
            [p.name for p in target_dir.iterdir()], [artifacts.METADATA_FILENAME]
        )

    def test_eval_only_run_with_stubbed_eval_still_writes_metadata(self):
        _, metadata, warnings = self.materialize(
            [], invocation={**INVOCATION, "mode": "eval"}
        )
        self.assertEqual(metadata["mode"], "eval")
        self.assertEqual(metadata["artifacts"], {})
        self.assertTrue(any("announced no artifacts" in w for w in warnings))


class RerunTests(MaterializeTestCase):
    def test_rerun_refreshes_link_without_duplicating_entries(self):
        first = self.native_file("first.json")
        second = self.native_file("second.json")

        self.materialize([f"CP_ARTIFACT inference {first}\n"])
        target_dir, metadata, _ = self.materialize(
            [f"CP_ARTIFACT inference {second}\n"]
        )
        record = metadata["artifacts"]["inference"]

        managed = sorted(
            p.name for p in target_dir.iterdir() if p.name.startswith("inference_output")
        )
        self.assertEqual(len(managed), 1)
        self.assertEqual(record["native_path"], str(second))
        self.assertEqual(
            self.resolve_link(target_dir, record), Path(os.path.realpath(second))
        )

    def test_rerun_preserves_created_at_and_advances_updated_at(self):
        native = self.native_file()
        line = f"CP_ARTIFACT inference {native}\n"
        _, first, _ = self.materialize([line])

        target_dir = artifacts.centralized_output_dir(
            self.outputs, "v1", "aut", "GPT4.1"
        )
        stored = json.loads(
            (target_dir / artifacts.METADATA_FILENAME).read_text(encoding="utf-8")
        )
        stored["created_at"] = "2000-01-01T00:00:00Z"
        (target_dir / artifacts.METADATA_FILENAME).write_text(
            json.dumps(stored), encoding="utf-8"
        )

        _, second, _ = self.materialize([line])
        self.assertEqual(second["created_at"], "2000-01-01T00:00:00Z")
        self.assertGreaterEqual(second["updated_at"], first["created_at"])

    def test_eval_only_rerun_keeps_the_earlier_inference_artifact(self):
        inference = self.native_file("inference.json")
        evaluation = self.native_file("eval.json")

        self.materialize(
            [f"CP_ARTIFACT inference {inference}\n"],
            invocation={**INVOCATION, "mode": "inference"},
        )
        target_dir, metadata, _ = self.materialize(
            [f"CP_ARTIFACT eval {evaluation}\n"],
            invocation={**INVOCATION, "mode": "eval"},
        )

        self.assertEqual(
            metadata["artifacts"]["inference"]["native_path"], str(inference)
        )
        self.assertEqual(metadata["artifacts"]["eval"]["native_path"], str(evaluation))
        self.assertEqual(
            self.resolve_link(target_dir, metadata["artifacts"]["inference"]),
            Path(os.path.realpath(inference)),
        )

    def test_stale_reference_file_is_cleared_when_symlinking_succeeds(self):
        native = self.native_file()
        target_dir = artifacts.centralized_output_dir(
            self.outputs, "v1", "aut", "GPT4.1"
        )
        target_dir.mkdir(parents=True)
        (target_dir / "inference_output.path").write_text("/stale\n", encoding="utf-8")

        _, metadata, _ = self.materialize([f"CP_ARTIFACT inference {native}\n"])
        record = metadata["artifacts"]["inference"]
        managed = sorted(
            p.name for p in target_dir.iterdir() if p.name.startswith("inference_output")
        )

        self.assertEqual(managed, [record["link"]])
        self.assertEqual(
            self.resolve_link(target_dir, record), Path(os.path.realpath(native))
        )


class AdapterContractTests(unittest.TestCase):
    ADAPTERS = ("aut", "ttcw", "creative_short", "ttct")
    # adapter name -> the `task` string the bundled evaluator dispatches on.
    BUNDLED_EVAL_TASKS = {
        "aut": "aut_push",
        "ttcw": "creative_writing",
        "creative_short": "creative_short",
    }

    def test_adapters_emit_markers_only_for_phases_they_run(self):
        for name in self.ADAPTERS:
            with self.subTest(adapter=name):
                source = (ADAPTER_DIR / f"{name}.sh").read_text(encoding="utf-8")
                self.assertIn('emit_artifact inference "$NATIVE_OUT"', source)
                self.assertNotIn("OUTPUT_PATH=", source)
                marker_line = next(
                    i for i, line in enumerate(source.splitlines())
                    if "emit_artifact inference" in line
                )
                guard_line = next(
                    i for i, line in enumerate(source.splitlines())
                    if '"$MODE" == "inference"' in line
                )
                self.assertLess(guard_line, marker_line)

    def test_adapters_emit_eval_markers_behind_an_eval_guard(self):
        for name in self.ADAPTERS:
            with self.subTest(adapter=name):
                lines = (ADAPTER_DIR / f"{name}.sh").read_text(
                    encoding="utf-8"
                ).splitlines()
                marker_line = next(
                    i for i, line in enumerate(lines) if "emit_artifact eval " in line
                )
                guard_line = next(
                    i for i, line in enumerate(lines) if '"$MODE" == "eval"' in line
                )
                inference_marker = next(
                    i for i, line in enumerate(lines)
                    if "emit_artifact inference " in line
                )
                self.assertLess(guard_line, marker_line)
                # The eval marker must live in its own branch, after the inference one.
                self.assertLess(inference_marker, guard_line)

    def test_bundled_eval_reuses_the_inference_run_id(self):
        """Eval reads data/output/<run_id>/inference_output.json, so the two configs
        must agree on run_id or evaluation silently targets the wrong directory."""
        for name, task_string in self.BUNDLED_EVAL_TASKS.items():
            with self.subTest(adapter=name):
                source = (ADAPTER_DIR / f"{name}.sh").read_text(encoding="utf-8")
                run_id_entry = f'"run_id": "${{RUN_ID}}/{name}/${{ALIAS}}"'
                self.assertEqual(source.count(run_id_entry), 2)
                self.assertIn(f'"task": "{task_string}"', source)
                self.assertIn('"model_name": "${JUDGE_ID}"', source)
                self.assertIn(
                    'emit_artifact eval "$NATIVE_OUT/eval_output_cleaned.json"', source
                )

    def test_ttct_eval_is_pinned_to_the_inference_run(self):
        source = (ADAPTER_DIR / "ttct.sh").read_text(encoding="utf-8")
        self.assertIn('-infer_model_name "$MODEL_SHORT"', source)
        self.assertIn('-eval_model_name "$JUDGE_ID"', source)
        self.assertIn('-run_id "$RUN_ID"', source)
        # Overrides a hardcoded site-specific default inside the task.
        self.assertIn('-api_key_path "$CREATIVITYPRISM_API_KEYS"', source)

    def test_adapters_never_copy_the_credentials_file(self):
        """Credentials are passed by path, never duplicated into the repo tree."""
        for name in self.ADAPTERS:
            with self.subTest(adapter=name):
                source = (ADAPTER_DIR / f"{name}.sh").read_text(encoding="utf-8")
                self.assertNotRegex(
                    source, r"\b(cp|mv|install|ln)\b[^\n]*CREATIVITYPRISM_API_KEYS"
                )

    def test_common_helper_rejects_unknown_kinds(self):
        source = (ADAPTER_DIR / "_common.sh").read_text(encoding="utf-8")
        self.assertIn("emit_artifact()", source)
        self.assertIn("inference|eval)", source)
        self.assertIn(f'echo "{artifacts.MARKER_PREFIX} ', source)


@unittest.skipUnless(shutil.which("bash"), "bash is required to run adapters")
class RunAdapterTests(unittest.TestCase):
    """The runner must echo adapter stdout live *and* keep it for parsing."""

    def run_script(self, body):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake_adapter.sh"
            script.write_text(body, encoding="utf-8")
            echoed = io.StringIO()
            with contextlib.redirect_stdout(echoed):
                rc, captured = runner_module.run_adapter(
                    ["bash", str(script)], os.environ.copy()
                )
            return rc, captured, echoed.getvalue()

    def test_stdout_is_both_echoed_and_captured(self):
        rc, captured, echoed = self.run_script(
            "echo starting\n"
            "echo 'CP_ARTIFACT inference /native/out.json'\n"
            "echo finished\n"
        )
        self.assertEqual(rc, 0)
        self.assertEqual(
            [line.strip() for line in captured],
            ["starting", "CP_ARTIFACT inference /native/out.json", "finished"],
        )
        self.assertIn("starting", echoed)
        self.assertIn("finished", echoed)
        self.assertEqual(
            artifacts.parse_artifact_markers(captured)[0],
            {"inference": "/native/out.json"},
        )

    def test_nonzero_exit_code_is_propagated_with_partial_output(self):
        rc, captured, _ = self.run_script("echo partial\nexit 7\n")
        self.assertEqual(rc, 7)
        self.assertEqual([line.strip() for line in captured], ["partial"])

    def test_undecodable_bytes_do_not_crash_the_runner(self):
        rc, captured, _ = self.run_script(
            "printf 'bad \\xff byte\\n'\n"
            "echo 'CP_ARTIFACT inference /native/out.json'\n"
        )
        self.assertEqual(rc, 0)
        self.assertEqual(
            artifacts.parse_artifact_markers(captured)[0],
            {"inference": "/native/out.json"},
        )


if __name__ == "__main__":
    unittest.main()
