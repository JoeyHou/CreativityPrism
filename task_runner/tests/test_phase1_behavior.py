import importlib.util
import contextlib
import copy
import io
import sys
import types
import re
import unittest
from pathlib import Path

from task_runner import run as runner_module


# task_runner/tests/ -> task_runner/ -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
from registry import read as _reg
# Same reason: the bundle's location is declared once, in registry/read.py.
INFERENCE_DIR = REPO_ROOT / _reg.BUILTIN_FOLDER / "src" / "inference"


class StubInferenceDriver:
    def __init__(self, config=None):
        self.config = config or {}


def make_prompt(data, template):
    return template.format(**data)


def install_import_stubs():
    modules = {
        "src": types.ModuleType("src"),
        "src.inference": types.ModuleType("src.inference"),
        "src.inference.inference_driver": types.ModuleType(
            "src.inference.inference_driver"
        ),
        "src.prompt_engineering": types.ModuleType("src.prompt_engineering"),
        "src.prompt_engineering.templates": types.ModuleType(
            "src.prompt_engineering.templates"
        ),
        "src.prompt_engineering.functions": types.ModuleType(
            "src.prompt_engineering.functions"
        ),
        "src.utils": types.ModuleType("src.utils"),
        "src.utils.helpers": types.ModuleType("src.utils.helpers"),
        "nltk": types.ModuleType("nltk"),
    }
    modules["src.inference.inference_driver"].InferenceDriver = StubInferenceDriver
    modules["src.prompt_engineering.templates"].conversation_history_template = (
        "[user]: {user_message}\n[assistant]: {assistant_message}\n"
    )
    modules["src.prompt_engineering.templates"].creative_writing_inference_template = (
        "Plot: {plot}\nWords: {word_count}"
    )
    modules["src.prompt_engineering.templates"].creative_short_story_template = (
        "Items: {items}\nTheme: {boring_theme}"
    )
    modules["src.prompt_engineering.functions"].make_prompt = make_prompt
    modules["src.utils.helpers"].load_json = lambda _path: {}
    modules["src.utils.helpers"].llm_batch_inference = lambda *_args, **_kwargs: []
    modules["nltk"].sent_tokenize = lambda text: [text]
    sys.modules.update(modules)


def load_inference_module(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, INFERENCE_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExactLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_import_stubs()
        cls.aut_module = load_inference_module("phase1_aut_push", "aut_push.py")
        cls.ttcw_module = load_inference_module(
            "phase1_creative_writing", "creative_writing.py"
        )
        cls.creative_short_module = load_inference_module(
            "phase1_creative_short", "creative_short_story.py"
        )

    @staticmethod
    def aut_data(size=5):
        return [
            {
                "meta_data": {"id": f"aut-{index}"},
                "input": {
                    "text": f"Use object {index}",
                    "others": {"prompt_type": "nc", "iteration_lst": {}},
                },
            }
            for index in range(size)
        ]

    @staticmethod
    def ttcw_data(size=5):
        return [
            {
                "meta_data": {"id": f"ttcw-{index}"},
                "input": {
                    "others": {
                        "plot": f"Plot {index} with {{{{word_count}}}} words",
                        "avg_len": 200,
                    }
                },
            }
            for index in range(size)
        ]

    @staticmethod
    def creative_short_data(size=5):
        return [
            {
                "meta_data": {"id": f"short-{index}"},
                "input": {
                    "others": {
                        "items": ["one", "two", str(index)],
                        "boring_theme": "an ordinary event",
                    }
                },
            }
            for index in range(size)
        ]

    def prompt_counts(self, limit):
        aut_driver = self.aut_module.AUTInference({"test_size": limit})
        ttcw_driver = self.ttcw_module.CreativeWritingInference({"test_size": limit})
        short_driver = self.creative_short_module.CreativeShortStoryInference(
            {"test_size": limit}
        )
        return {
            "aut": len(
                aut_driver.create_batched_prompt(self.aut_data(), {}, curr_iter="")
            ),
            "ttcw": len(ttcw_driver.create_batched_prompt(self.ttcw_data())),
            "creative_short": len(
                short_driver.create_batched_prompt(self.creative_short_data())
            ),
        }

    def test_positive_limits_are_exact(self):
        for limit in (1, 3):
            with self.subTest(limit=limit):
                self.assertEqual(
                    self.prompt_counts(limit),
                    {"aut": limit, "ttcw": limit, "creative_short": limit},
                )

    def test_limit_above_dataset_size_returns_available_samples(self):
        self.assertEqual(
            self.prompt_counts(10),
            {"aut": 5, "ttcw": 5, "creative_short": 5},
        )

    def test_negative_one_preserves_unlimited_behavior(self):
        self.assertEqual(
            self.prompt_counts(-1),
            {"aut": 5, "ttcw": 5, "creative_short": 5},
        )


class NativeOutputIsolationTests(unittest.TestCase):
    """Two tasks must never write into the same native output directory.

    The adapters used to spell the path inline, so this could be a string match.
    They now build it from a RUN_PATH variable, and the eight of them share one
    shape -- which is exactly why a collision would be easy to introduce and
    hard to see. Assert the property (all distinct), not the spelling.
    """

    # Only the built-in bundle. RUN_PATH is one of its internal conventions;
    # a contributed task decides its own output layout, and asserting ours
    # against it would be a tax disguised as a gate.
    from registry import read as _registry_read
    ADAPTERS = tuple(sorted(_registry_read.builtin_tasks()))

    def test_bundled_adapters_define_distinct_task_paths(self):
        paths = {}
        for task_name in self.ADAPTERS:
            adapter = REPO_ROOT / "registry" / "adapters" / f"{task_name}.sh"
            source = adapter.read_text(encoding="utf-8")
            match = re.search(r'^RUN_PATH="([^"]+)"', source, re.M)
            self.assertIsNotNone(match, f"{task_name}.sh defines no RUN_PATH")
            run_path = match.group(1)
            # The task segment is what keeps two tasks apart under one run id.
            self.assertIn(f"/{task_name}/", run_path,
                          f"{task_name}.sh RUN_PATH lacks its own task segment")
            self.assertIn('data/output/${RUN_PATH}', source)
            paths[task_name] = run_path

        self.assertEqual(len(set(paths.values())), len(self.ADAPTERS),
                         f"colliding native output paths: {paths}")


class LimitValidationTests(unittest.TestCase):
    def setUp(self):
        self.tasks = {
            "supported": {"name": "supported", "limit_supported": True},
            "unsupported": {"name": "unsupported", "limit_supported": False},
        }
        self.models = {"inference": {}, "judge": {}}

    def config(self, task="supported", limit=None):
        return {
            "task": task,
            "inference_model": "inference",
            "judge_model": "judge",
            "limit": limit,
        }

    def assert_validation_error(self, config, message):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                runner_module.validate_run_config(
                    config, self.tasks, self.models, source_label="test"
                )
        self.assertEqual(raised.exception.code, 5)
        self.assertIn(message, stderr.getvalue())

    def test_positive_limit_is_accepted(self):
        runner_module.validate_run_config(
            self.config(limit=1), self.tasks, self.models, source_label="test"
        )

    def test_non_positive_and_non_integer_limits_are_rejected(self):
        for limit in (0, -1, "3", True):
            with self.subTest(limit=limit):
                self.assert_validation_error(
                    self.config(limit=limit), "'limit' must be a positive integer"
                )

    def test_limit_is_rejected_for_unsupported_task(self):
        self.assert_validation_error(
            self.config(task="unsupported", limit=1),
            "limit is not supported by task 'unsupported'",
        )

    def test_cli_limit_uses_shared_validation(self):
        args = types.SimpleNamespace(
            config=None,
            task="supported",
            model="inference",
            judge_model="judge",
            label="run",
            run_id=None,
            limit=0,
        )
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                runner_module.resolve_invocation(args, self.tasks, self.models)
        self.assertEqual(raised.exception.code, 5)
        self.assertIn("'limit' must be a positive integer", stderr.getvalue())


class TTCTExactLimitTests(unittest.TestCase):
    """--limit caps TTCT *per question type*, and never shrinks the row count.

    Both halves matter and they pull in opposite directions. The cap is per type
    because that is how the original counted, so `--limit 2` over 5 types is 10
    prompts, not 2 -- a reading that silently divides the sample size by 5 if
    someone "fixes" it. And the surplus rows stay in the output marked skip
    rather than being dropped, because evaluation walks the full row set.

    Written against tasks/prism_v1's driver. The archived bundle's ttct_inference.main
    is gone, so this exercises create_batched_prompt directly, which is where the
    cap actually lives.
    """

    QUESTION_TYPES = ("1_unusual_uses", "2_consequences")

    @classmethod
    def setUpClass(cls):
        # Same stubs and same file-path load as the classes above: the stubbed
        # "src.*" modules are already in sys.modules by now, so a plain import
        # would resolve to those rather than to the real package.
        install_import_stubs()
        cls.driver_cls = load_inference_module("phase1_ttct", "ttct.py").TTCTInference
        cls.dataset = [
            {
                "meta_data": {"id": f"ttct-{qt}-{i}", "question_type": qt},
                "input": {"text_cot": f"cot-{qt}-{i}",
                          "text_basic": f"basic-{qt}-{i}",
                          "text_instructive": f"instructive-{qt}-{i}"},
            }
            for qt in cls.QUESTION_TYPES
            for i in range(5)
        ]

    def prompts_for(self, num_samples):
        # No model, no network: build the driver without __init__ and give it the
        # only two attributes create_batched_prompt reads.
        driver = self.driver_cls.__new__(self.driver_cls)
        driver.config = {"num_samples": num_samples, "subset": list(self.QUESTION_TYPES)}
        return driver.create_batched_prompt(self.dataset, "cot")

    def test_limit_caps_each_question_type_not_the_total(self):
        for limit, expected in ((1, 2), (3, 6), (10, 10)):
            with self.subTest(limit=limit):
                prompts = self.prompts_for(limit)
                self.assertEqual(len(prompts), expected)
                per_type = {}
                for p in prompts:
                    qt = p["data"]["meta_data"]["question_type"]
                    per_type[qt] = per_type.get(qt, 0) + 1
                for qt in self.QUESTION_TYPES:
                    self.assertLessEqual(per_type.get(qt, 0), limit)

    def test_negative_one_means_unlimited(self):
        self.assertEqual(len(self.prompts_for(-1)), len(self.dataset))

    def test_capped_rows_keep_their_dataset_index(self):
        # Evaluation walks every row and reads `row_index` back, so a cap that
        # renumbered the survivors would misalign scores against generations.
        prompts = self.prompts_for(3)
        self.assertEqual([p["row_index"] for p in prompts], [0, 1, 2, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()