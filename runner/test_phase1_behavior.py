import importlib.util
import contextlib
import copy
import io
import sys
import types
import unittest
from pathlib import Path

from runner import run as runner_module


REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = REPO_ROOT / "tasks" / "aut_ttcw_cshort" / "src" / "inference"


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
    def test_bundled_adapters_define_distinct_task_paths(self):
        adapters = {
            "aut": REPO_ROOT / "registry" / "adapters" / "aut.sh",
            "ttcw": REPO_ROOT / "registry" / "adapters" / "ttcw.sh",
            "creative_short": (
                REPO_ROOT / "registry" / "adapters" / "creative_short.sh"
            ),
        }

        output_templates = set()
        for task_name, adapter in adapters.items():
            source = adapter.read_text(encoding="utf-8")
            logical_run_id = f'"run_id": "${{RUN_ID}}/{task_name}/${{ALIAS}}"'
            output_template = f'data/output/${{RUN_ID}}/{task_name}/${{ALIAS}}'
            self.assertIn(logical_run_id, source)
            self.assertIn(output_template, source)
            output_templates.add(output_template)

        self.assertEqual(len(output_templates), len(adapters))


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
    @classmethod
    def setUpClass(cls):
        install_import_stubs()
        cls.dataset = [
            {"meta_data": {"id": f"ttct-{index}", "question_type": "1_unusual_uses"}}
            for index in range(5)
        ]
        cls.saved_data = []

        helpers = sys.modules["src.utils.helpers"]

        def load_json(path):
            if path == "dataset.json":
                return copy.deepcopy(cls.dataset)
            return {"gpt-test": "unused"}

        def load_prompts_as_list(data_path, prompt_type, subset):
            return [f"{prompt_type}-{index}" for index in range(len(cls.dataset))]

        def save_json(data, _path):
            cls.saved_data = copy.deepcopy(data)

        helpers.load_json = load_json
        helpers.load_prompts_as_list = load_prompts_as_list
        helpers.save_json = save_json

        api_wrapper = types.ModuleType("src.utils.api_wrapper")

        class StubModelWrapper:
            def __init__(self, model_name, api_key):
                self.model_name = model_name
                self.api_key = api_key

            def generate_response(self, prompt):
                return f"response:{prompt}"

        api_wrapper.ModelWrapper = StubModelWrapper
        sys.modules["src.utils.api_wrapper"] = api_wrapper

        tqdm_module = types.ModuleType("tqdm")
        tqdm_module.tqdm = lambda iterable: iterable
        sys.modules["tqdm"] = tqdm_module

        cls.ttct_module = load_inference_module(
            "phase1_ttct", "../../../ttct/src/inference/ttct_inference.py"
        )

    def run_ttct(self, limit):
        with contextlib.redirect_stdout(io.StringIO()):
            self.ttct_module.main(
                model_name="gpt-test",
                temp=1,
                cache_dir="",
                subset=self.ttct_module.DEFAULT_SUBSET,
                prompt_formats="all",
                api_key_path="keys.json",
                data_path="dataset.json",
                run_id="test",
                num_samples=limit,
            )
        return self.saved_data

    def test_positive_limits_are_exact_and_lists_remain_aligned(self):
        for limit, expected in ((1, 1), (3, 3), (10, 5)):
            with self.subTest(limit=limit):
                result = self.run_ttct(limit)
                self.assertEqual(len(result), expected)
                for item in result:
                    self.assertEqual(
                        set(item["output"]),
                        {"text_basic", "text_instructive", "text_cot"},
                    )

    def test_negative_one_preserves_internal_unlimited_behavior(self):
        self.assertEqual(len(self.run_ttct(-1)), len(self.dataset))


if __name__ == "__main__":
    unittest.main()