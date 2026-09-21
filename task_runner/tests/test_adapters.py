"""Gate: every registered adapter can be driven by the runner.

No GPU, no API key, no paid call -- CREATIVITYPRISM_ADAPTER_DRYRUN makes each
adapter print what it would have run instead of running it.

The assertions come in two kinds, and keeping them apart is the point:

* **Contract** applies to every task, contributed or built-in. An adapter must
  offer both stages, must honour its registry entry's judge policy, and must not
  duplicate credentials.
* **Convention** applies only to tasks whose folder is the built-in bundle. Its
  entrypoints take a JSON config, name one run_id variable, and write
  inference_output.json -- all worth asserting for it, all wrong to demand of a
  contributed task that keeps its own layout.

Asserting the second kind against everyone is a tax dressed up as a gate: it was
what made a deliberately messy contributed task fail three gates for reasons
that had nothing to do with whether it worked.
"""
import json
import os
import re
import subprocess
import unittest

from registry import read as registry_read

# task_runner/tests/ -> task_runner/ -> repo root.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADAPTERS = registry_read.task_names()
BUILTIN = sorted(registry_read.builtin_tasks())
JUDGE_TASKS = registry_read.judge_tasks()
JUDGE_CANONICAL = "GPT4.1-mini"


def dispatch_keys(entrypoint, name):
    # Folder from the registry, not spelled here: renaming the bundle should
    # touch one constant, not every file that happens to know where it lives.
    with open(os.path.join(REPO, registry_read.BUILTIN_FOLDER, entrypoint)) as f:
        src = f.read()
    block = re.search(re.escape(name) + r"\s*=\s*\{(.*?)\n\}", src, re.S).group(1)
    return set(re.findall(r"'([^']+)':", block))


def run_adapter(slug, model="Qwen2.5-7B", judge="GPT4.1-mini"):
    """Dry-run one adapter. Returns (config_stages, command_stages)."""
    env = dict(os.environ, CREATIVITYPRISM_ADAPTER_DRYRUN="1")
    out = subprocess.run(
        ["bash", os.path.join(REPO, "registry/adapters", slug + ".sh"),
         "--model", model, "--judge-model", judge,
         "--run-id", "gate", "--output-dir", "/tmp/cp_gate"],
        capture_output=True, text=True, env=env, timeout=180,
    )
    if out.returncode != 0:
        raise AssertionError("{}.sh rc={}: {}".format(
            slug, out.returncode, (out.stderr or out.stdout)[-500:]))
    configs, commands = {}, []
    for block in out.stdout.split("### "):
        name, _, rest = block.partition("\n")
        name = name.strip()
        body = rest.split("### END")[0]
        if name in ("run_inference.py", "run_evaluation.py"):
            configs[name] = json.loads(body)
        elif name == "CMD":
            commands.append(body.strip())
    return configs, commands


class AdapterContract(unittest.TestCase):
    """True of every registered task."""

    @classmethod
    def setUpClass(cls):
        cls.runs = {s: run_adapter(s) for s in ADAPTERS}

    def test_adapter_offers_the_stages_it_declares(self):
        # Not "two": a generation-only task scored by hand later is a real shape,
        # and demanding an evaluation stage from it would be the gate inventing a
        # requirement. The registry says how many; either output shape counts.
        for slug in ADAPTERS:
            configs, commands = self.runs[slug]
            expected = len(registry_read.stages(slug))
            with self.subTest(slug):
                self.assertGreaterEqual(
                    len(configs) + len(commands), expected,
                    "registry declares stages %r" % (registry_read.stages(slug),))

    def test_judge_policy_matches_the_registry(self):
        """uses_judge: false means --judge-model really is a no-op.

        Checked by name rather than by shape, because the two adapter styles pass
        the judge differently -- a config field for the built-in bundle, a CLI
        flag for a contributed task -- and because either the canonical name or
        its alias may be what reaches the task. Hardcoding one spelling here is
        how this test first failed against a task that was behaving correctly.
        """
        import yaml
        with open(os.path.join(REPO, "registry/models.yaml")) as f:
            models = yaml.safe_load(f)["models"]
        judge_names = {JUDGE_CANONICAL}
        alias = (models.get(JUDGE_CANONICAL, {}).get("aliases") or {}).get("all")
        if alias:
            judge_names.add(alias)

        for slug in ADAPTERS:
            configs, commands = self.runs[slug]
            eval_cfg = configs.get("run_evaluation.py")
            haystack = json.dumps(eval_cfg or {}) + "\n".join(commands)
            # The inference stage legitimately names the *inference* model; only
            # the eval side is evidence about the judge.
            if eval_cfg is None and commands:
                haystack = "\n".join(commands[1:]) if len(commands) > 1 else ""
            named = any(n in haystack for n in judge_names)
            with self.subTest(slug):
                self.assertEqual(
                    named, slug in JUDGE_TASKS,
                    "registry says uses_judge=%s but the eval stage %s the judge"
                    % (slug in JUDGE_TASKS, "names" if named else "does not name"))

    def test_no_adapter_duplicates_credentials(self):
        for slug in ADAPTERS:
            with self.subTest(slug):
                with open(os.path.join(REPO, "registry/adapters", slug + ".sh")) as f:
                    source = f.read()
                self.assertNotRegex(source, r"(cp|copy|install)\s+[^\n]*api_keys\.json")


class BuiltinBundleConventions(unittest.TestCase):
    """True only of tasks implemented inside tasks/prism_v1."""

    @classmethod
    def setUpClass(cls):
        cls.inference_tasks = dispatch_keys("run_inference.py", "INFERENCE_DRIVERS")
        cls.eval_tasks = dispatch_keys("run_evaluation.py", "EVAL_DRIVERS")
        cls.runs = {s: run_adapter(s)[0] for s in BUILTIN}

    def test_both_stages_emit_a_config(self):
        for slug in BUILTIN:
            with self.subTest(slug):
                self.assertEqual(set(self.runs[slug]),
                                 {"run_inference.py", "run_evaluation.py"})

    def test_task_names_are_dispatchable(self):
        for slug in BUILTIN:
            for script, known in (("run_inference.py", self.inference_tasks),
                                  ("run_evaluation.py", self.eval_tasks)):
                with self.subTest(slug=slug, script=script):
                    task = self.runs[slug][script]["experiments_list"][0]["task"]
                    self.assertIn(task, known)

    def test_both_stages_share_a_run_id(self):
        for slug in BUILTIN:
            with self.subTest(slug):
                a = self.runs[slug]["run_inference.py"]["experiments_list"][0]["run_id"]
                b = self.runs[slug]["run_evaluation.py"]["experiments_list"][0]["run_id"]
                self.assertEqual(a, b)

    def test_native_output_paths_are_distinct(self):
        paths = {s: self.runs[s]["run_inference.py"]["experiments_list"][0]["run_id"]
                 for s in BUILTIN}
        self.assertEqual(len(set(paths.values())), len(BUILTIN), paths)

    def test_eval_input_points_inside_this_run(self):
        for slug in BUILTIN:
            e = self.runs[slug]["run_evaluation.py"]["experiments_list"][0]
            if "eval_input_dir" not in e:
                continue
            with self.subTest(slug):
                self.assertTrue(
                    e["eval_input_dir"].startswith("data/output/" + e["run_id"]),
                    (slug, e["eval_input_dir"], e["run_id"]))


class RegistryConsistency(unittest.TestCase):

    def test_every_registry_model_resolves(self):
        import yaml
        with open(os.path.join(REPO, "registry/models.yaml")) as f:
            models = yaml.safe_load(f)["models"]
        with open(os.path.join(REPO, registry_read.BUILTIN_FOLDER, "src/driver.py")) as f:
            src = f.read()
        body = re.search(r"self\.open_source_models\s*=\s*\{(.*?)\n        \}", src, re.S).group(1)
        open_keys = set(re.findall(r'^\s+"([^"]+)":\s*\{', body, re.M))
        api_path = os.environ.get("CREATIVITYPRISM_API_KEYS",
                                  os.path.join(REPO, "api_keys.json"))
        api_keys = None
        if os.path.isfile(api_path):
            # Names only. The values are credentials and are never read here.
            with open(api_path) as f:
                api_keys = set(json.load(f))
        for name, spec in models.items():
            alias = (spec.get("aliases") or {}).get("all")
            with self.subTest(name):
                self.assertIsNotNone(alias, "{} has no 'all' alias".format(name))
                if spec["type"] == "open":
                    self.assertIn(alias, open_keys)
                elif api_keys is not None:
                    self.assertIn(alias, api_keys)

    def test_every_task_declares_a_resolvable_parser(self):
        # A task with no working parser is invisible in the shared table, which
        # is a silent way to not be in the benchmark.
        parsers = registry_read.parsers()
        for name, data in registry_read.load_tasks().items():
            with self.subTest(name):
                self.assertIn("loader_parser", data,
                              "declare loader_parser so results reach the table")
                self.assertIn(name, parsers,
                              "loader_parser %r did not import" % data.get("loader_parser"))

    def test_every_adapter_on_disk_is_registered(self):
        on_disk = sorted(
            f[:-3] for f in os.listdir(os.path.join(REPO, "registry/adapters"))
            if f.endswith(".sh") and not f.startswith("_")
        )
        self.assertEqual(on_disk, sorted(ADAPTERS))


if __name__ == "__main__":
    unittest.main()
