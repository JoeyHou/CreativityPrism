"""NeoCoder evaluation: correctness, technique detection, and the aggregate.

Reimplements the correctness half of ``neocoder_dat/src/evaluators/
dp_evaluator.py`` (original bundle, kept locally under tasks/archive/ and not shipped with the repo). The execution machinery -- subprocess isolation, the timeout,
the type-agnostic output comparison -- is imported unchanged from
``neocoder_exec`` (a copy of the original ``evaluation_utils``). That code is
what decides whether a solution counts as correct, and rewriting it would put
every published Convergence number in question for no gain.

**This runs model-generated code.** Each attempt is a subprocess in a temporary
directory with its own timeout, which is how HumanEval and every other execution
benchmark does it, but it is still untrusted code executing under your account.
Know that before pointing it at a large sweep.

**Three stages, selected by ``stage`` in the config.** ``correctness`` executes
the code; ``techniques`` asks a reviewer model which of 34 named techniques each
solution used; ``creativity`` joins the two into Convergence and Divergence.

One thing to know before reporting anything called "@5": the aggregate drops
rounds whose constraint set equals the previous round's, and the shipped dataset
does repeat them -- 1901A has four distinct sets across six rounds -- so the
denial index is not the round index.

Correctness is reported per round and, per the paper's Convergence@5, at the last
round as well. Rounds are also counted by *why* they failed: not parsable, not
executable, timed out, or wrong answer. All four were a bare ``False`` in the
original, which is how "every model scores 0.0000" reads as a result rather than
as a question.
"""
import re

import pandas as pd

from src.evaluation.eval_driver import EvalDriver
from src.evaluation.neocoder_exec import (
    function_with_timeout,
    run_solution,
    type_agnostic_compare,
)
from src.utils.helpers import data_path, load_json
from src.utils.api_wrapper import ModelWrapper
from src.prompt_engineering.neocoder_techniques import (
    CODE_REVIEWER,
    MAX_PLAUSIBLE_TECHNIQUES,
    parse_techniques,
)

SOLUTION_TIMEOUT = 6.0

NOT_PARSABLE = "code not parsable"
NOT_EXECUTABLE = "code not executable"
TIMED_OUT = "code execution timeout"
WRONG = "wrong answer"
CORRECT = "correct"


def _normalise_stdin(code):
    for old in ("sys.stdin.read()", "stdin.read()", "sys.stdin.readlines()",
                "stdin.readlines()", "sys.stdin.readline()", "stdin.readline()"):
        code = code.replace(old, "input()")
    return code


def parse_code_fenced(text):
    """Take the first ```python block, which is where the model put the code.

    This is not a new idea: the *generator* already extracted code this way
    (``CodeGenerator.parse_response``). Only the evaluator used the substring
    walk below, so one pipeline shipped two extractors and scored with the worse
    one.
    """
    if text is None:
        return None
    match = re.search(r"```(?:python|Python)[^\n]*\n(.*?)```", text, re.S)
    if match is None:
        return None
    return _normalise_stdin(match.group(1))


def parse_code(code):
    """Lifted from CodeForceCorrectnessEvaluator.parse_code, unchanged in effect.

    Rewrites the various stdin idioms to ``input()``, finds the first ``def``,
    walks back to the imports above it, and keeps the body of ``solve()``.

    **This discards most valid solutions, and the reason is substring matching.**
    ``code.find("def")``, ``find("import")`` and ``find("from")`` run over the
    whole response, prose included. Measured over 24 generations (12 gpt-4.1-mini,
    12 Qwen2.5-7B):

    * **19 of 24 fall through to ``return None``**, and all 19 for the same
      reason: ``import_idx == -1`` while ``from_idx != -1``, because the English
      word "from" appears in the model's explanation -- "the trip *from* 0 to x",
      "subtract 1 *from* the integer". No branch covers that combination.
    * 3 more contain no ``def`` at all.
    * 2 parse.

    A curiosity that is *not* the cause, since it happened once and that response
    also had a prose "from": ``find("def")`` can match inside **"Codeforces"**
    (Co-*def*-orces), the name of the dataset's own source, and one gpt-4.1-mini
    answer said "the problem is from Codeforces".

    Every one of the 22 discarded responses was scored "code not parsable", i.e.
    incorrect. That is the shape of a table where Convergence@5 is 0.0000 for
    every model.

    Kept because it is what produced the published numbers. Choose with
    ``code_extractor`` in the config.
    """
    if code is None:
        return None

    code = _normalise_stdin(code)

    def_idx = code.find("def")
    if def_idx == -1:
        return None

    import_idx = code.find("import", 0, def_idx)
    from_idx = code.find("from", 0, def_idx)
    if import_idx == -1 and from_idx == -1:
        code = code[def_idx:]
    elif import_idx > from_idx and from_idx != -1:
        code = code[from_idx:]
    elif import_idx != -1 and from_idx == -1:
        code = code[import_idx:]
    elif import_idx != -1 and from_idx > import_idx:
        code = code[import_idx:]
    else:
        return None

    solve_idx = code.find("solve(")
    prefix_code = code[:solve_idx]
    for line in code[solve_idx:].split("\n"):
        if line.startswith(" ") or line.startswith("\t") or line.startswith("solve"):
            prefix_code += line + "\n"
        else:
            break
    return prefix_code


def test_correctness(code, test_case_inputs, test_case_outputs):
    """(all_passed, outputs) or (False, None) when the code will not compile."""
    try:
        compile(code, "<solution>", "exec")
    except SyntaxError:
        return False, None

    # First try: every case at once, the way a Codeforces multi-test file is laid out.
    test_input = [" ".join(row) for case in test_case_inputs for row in case]
    test_input.insert(0, str(len(test_case_inputs)))
    stdout = run_solution(code, test_input)
    if stdout is not None:
        try:
            output = stdout.strip().split("\n")
            passed = [type_agnostic_compare(o, e)
                      for o, e in zip(output, test_case_outputs)]
            return all(passed), output
        except Exception:
            # type_agnostic_compare eval()s stdout when the expectation is a
            # list, so a solution that prints prose lands here.
            pass

    # Second try: one case at a time.
    output, passed = [], []
    for case_in, case_out in zip(test_case_inputs, test_case_outputs):
        stdout = run_solution(code, [" ".join(row) for row in case_in])
        if stdout is None:
            passed.append(False)
            output.append(None)
            continue
        try:
            text = stdout.strip()
            passed.append(type_agnostic_compare(text, case_out))
            output.append(text)
        except Exception:
            passed.append(False)
            output.append(None)
    return all(passed), output


class NeoCoderEval(EvalDriver):

    def __init__(self, config={}):
        super().__init__(config)
        self.test_case_path = config.get(
            "test_case_path", data_path("neocoder/test_cases_annotated.json"))

        self.code_extractor = config.get("code_extractor")
        if self.code_extractor not in ("legacy", "fenced"):
            raise ValueError(
                "neocoder needs an explicit 'code_extractor'. 'legacy' is the "
                "substring walk that produced the published numbers and that "
                "discards most valid solutions (see parse_code); 'fenced' takes "
                "the ```python block, which is what the generator side already "
                "did. They do not give the same Convergence."
            )
        self._extract = parse_code if self.code_extractor == "legacy" else parse_code_fenced

        self.stage = config.get("stage", "correctness")
        if self.stage not in ("correctness", "techniques", "creativity"):
            raise ValueError(
                "neocoder 'stage' must be correctness, techniques or creativity; "
                "got {!r}".format(self.stage)
            )
        self.human_techniques_path = config.get(
            "human_techniques_path",
            data_path("neocoder/human_solution_techniques.json"))

    def _test_cases(self):
        cases = {}
        for example in load_json(self.test_case_path):
            problem_id = example["problem_id"]
            if problem_id in cases:
                raise ValueError("duplicate problem_id in test cases: {}".format(problem_id))
            cases[problem_id] = {"input": example["input"], "output": example["output"]}
        return cases

    def create_batched_prompt(self, input_data):
        """One execution job per (problem, round). No prompts involved."""
        jobs = []
        for problem in input_data:
            for dp_idx, output in enumerate(problem.get("outputs", [])):
                jobs.append({
                    "problem_id": problem["problem_id"],
                    "dp_idx": dp_idx,
                    "raw_output": output,
                    "data": problem,
                })
        return jobs

    def parse_llm_outputs(self, llm_outputs):
        for job in llm_outputs:
            job["code"] = self._extract(job["raw_output"])
        return llm_outputs

    def _run_one(self, code, case):
        if code is None:
            return False, NOT_PARSABLE
        try:
            # Backstop only: each attempt is a subprocess enforcing its own
            # timeout, and one retry per case needs more wall clock than a flat cap.
            correct, output = function_with_timeout(
                test_correctness,
                (code, case["input"], case["output"]),
                timeout=SOLUTION_TIMEOUT * (len(case["input"]) + 1) + 5,
            )
        except TimeoutError:
            return False, TIMED_OUT
        if output is None:
            return bool(correct), NOT_EXECUTABLE
        return bool(correct), (CORRECT if correct else WRONG)

    def evaluation(self):
        input_data = load_json(self.config["eval_input_dir"])
        if not input_data:
            raise FileNotFoundError(
                "NeoCoder inference output not found: {}".format(
                    self.config["eval_input_dir"]))

        if self.stage == "techniques":
            return self._run_techniques(input_data)
        if self.stage == "creativity":
            return self._run_creativity(input_data)

        cases = self._test_cases()
        jobs = self.parse_llm_outputs(self.create_batched_prompt(input_data))
        print("=> executing {} generated solution(s)".format(len(jobs)))

        eval_output_cleaned = []
        for job in jobs:
            case = cases.get(job["problem_id"])
            if case is None:
                correct, reason = False, "no test case"
            else:
                correct, reason = self._run_one(job["code"], case)
            eval_output_cleaned.append({
                "meta_data": job["data"].get("meta_data", {}),
                "problem_id": job["problem_id"],
                "dp_idx": job["dp_idx"],
                "constraints": (job["data"].get("constraints") or [None] * (job["dp_idx"] + 1))[job["dp_idx"]]
                if job["dp_idx"] < len(job["data"].get("constraints") or []) else None,
                "correctness": correct,
                # The original recorded a bare False for all four failure modes.
                "reason": reason,
                "code_extractor": self.code_extractor,
            })
            print("   {} dp{} -> {}".format(job["problem_id"], job["dp_idx"], reason), flush=True)

        return self.generate_eval_report(eval_output_cleaned), eval_output_cleaned

    # ---------- stage: techniques ----------

    def _review_techniques(self, reviewer, code, max_attempts=3):
        """Which of the 34 techniques this code uses, or None if unreadable.

        The original looped ``while True`` until the reviewer named 10 or fewer,
        with no bound: a model that keeps echoing the prompt's own list spends
        money forever. Bounded here, and a run that never converges returns None
        rather than a guess.
        """
        for _ in range(max_attempts):
            reply = reviewer.generate_response(
                [{"role": "system", "content": CODE_REVIEWER},
                 {"role": "user", "content": code}],
                {"max_tokens": 256, "temperature": 0.75},
            )
            if isinstance(reply, str) and "Error" in reply[:40]:
                return None
            found = parse_techniques(reply)
            # More than this and the reviewer has echoed the list, not read the code.
            if len(found) <= MAX_PLAUSIBLE_TECHNIQUES:
                return found
        return None

    def _run_techniques(self, input_data):
        reviewer_name = self.config.get("reviewer_model", "gpt-4-turbo")
        cred = self.config.get("reviewer_credential", reviewer_name)
        if cred not in self.api_models:
            raise ValueError(
                "no credentials for reviewer {!r}. Present: {}".format(
                    cred, sorted(self.api_models)))
        reviewer = ModelWrapper(reviewer_name, self.api_models[cred])

        print("=> detecting techniques with {} over {} problem(s)".format(
            reviewer_name, len(input_data)))

        rows = []
        for item in input_data:
            per_round = []
            for dp_idx, output in enumerate(item.get("outputs", [])):
                code = self._extract(output)
                found = self._review_techniques(reviewer, code) if code else None
                per_round.append(found)
                print("   {} dp{} -> {}".format(
                    item["problem_id"], dp_idx,
                    "unreadable" if found is None else ", ".join(found) or "none"),
                    flush=True)
            rows.append({
                "meta_data": item.get("meta_data", {}),
                "problem_id": item["problem_id"],
                "constraints": item.get("constraints", []),
                # None means the reviewer could not be read, which is different
                # from "used no technique". The aggregate keeps them apart.
                "techniques": per_round,
                "reviewer_model": reviewer_name,
                "code_extractor": self.code_extractor,
            })
        return self._techniques_report(rows), rows

    @staticmethod
    def _techniques_report(rows):
        flat = [(r["problem_id"], i, t)
                for r in rows for i, t in enumerate(r["techniques"])]
        return pd.DataFrame([{
            "n_rounds": len(flat),
            "n_unreadable": sum(1 for _, _, t in flat if t is None),
            "n_empty": sum(1 for _, _, t in flat if t == []),
            "mean_techniques": (
                sum(len(t) for _, _, t in flat if t) / max(1, sum(1 for _, _, t in flat if t))
            ),
        }])

    # ---------- stage: creativity ----------

    def _run_creativity(self, _unused):
        """Join correctness and techniques into Convergence and Divergence.

        Rounds whose constraint set repeats the previous round's are dropped,
        which is what makes the denial index differ from the round index. Kept
        because it is the published behaviour, and surfaced as ``denial_index``
        next to ``dp_idx`` so the difference is visible rather than implied.
        """
        correctness = load_json(self.config["correctness_path"])
        techniques = load_json(self.config["techniques_path"])
        human = load_json(self.human_techniques_path)
        if not correctness or not techniques:
            raise FileNotFoundError(
                "creativity needs both correctness_path and techniques_path")

        correct_by = {(r["problem_id"], r["dp_idx"]): r["correctness"]
                      for r in correctness}
        rows = []
        for rec in techniques:
            pid = rec["problem_id"]
            human_set = {tech for sol in human.get(pid, []) for tech in sol}
            prev_constraint, denial_index = None, 0
            for dp_idx, found in enumerate(rec["techniques"]):
                constraints = rec.get("constraints") or []
                constraint = constraints[dp_idx] if dp_idx < len(constraints) else None
                if constraint == prev_constraint:
                    continue
                prev_constraint = constraint
                novel = None if found is None else [t for t in found if t not in human_set]
                rows.append({
                    "problem_id": pid,
                    "dp_idx": dp_idx,
                    "denial_index": denial_index,
                    "constraint": constraint,
                    "correct": correct_by.get((pid, dp_idx)),
                    "techniques": found,
                    "novel_techniques": novel,
                    "divergence": (None if not found else len(novel) / len(found)),
                    "human_techniques_known": bool(human_set),
                })
                denial_index += 1
        return self._creativity_report(rows), rows

    @staticmethod
    def _creativity_report(rows):
        if not rows:
            return pd.DataFrame(columns=["denial_index", "n", "convergence", "divergence"])
        df = pd.DataFrame(rows)
        out = []
        for idx, group in df.groupby("denial_index"):
            scored = group[group["divergence"].notna()]
            out.append({
                "denial_index": idx,
                "n": len(group),
                "convergence": group["correct"].fillna(False).mean(),
                "divergence": scored["divergence"].mean() if len(scored) else None,
                # A divergence averaged over rounds whose techniques could not be
                # read is not the published quantity; say how many those were.
                "n_unreadable": int(group["techniques"].isna().sum()),
            })
        return pd.DataFrame(out)

    def generate_eval_report(self, eval_outputs):
        """Convergence per round, plus why the failures failed."""
        if not eval_outputs:
            return pd.DataFrame(columns=["dp_idx", "n", "n_correct", "convergence"])

        df = pd.DataFrame(eval_outputs)
        report = (
            df.groupby("dp_idx")
            .agg(n=("correctness", "size"), n_correct=("correctness", "sum"))
            .reset_index()
        )
        report["convergence"] = report["n_correct"] / report["n"]
        for reason in (NOT_PARSABLE, NOT_EXECUTABLE, TIMED_OUT, WRONG):
            counts = df[df["reason"] == reason].groupby("dp_idx").size()
            report[reason.replace(" ", "_")] = report["dp_idx"].map(counts).fillna(0).astype(int)
        return report
