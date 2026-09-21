"""Gate for the judge-verdict parsers. No network, no model, no API key.

Every task here ends the same way: a judge writes prose and something has to
read a verdict out of it. That step has been the quietest source of wrong
numbers in this benchmark, because when it fails it does not raise -- it returns
the "no" value, and a broken parser is indistinguishable from a harsh judge.

The cases below are real judge replies from run b1 (2026-09-03), not invented
ones. The TTCW case is the bug that motivated the file: the binary branch read
the first 20 characters of a reply whose verdict the prompt puts *last*, so over
30 replies it returned 0 every single time, including the 4 that said Yes.
"""
import unittest

from src.evaluation.creative_writing import CreativeWritingEval
from src.evaluation.creative_math import extract_yes_no


def _clean(raws, five_scale=False):
    ev = CreativeWritingEval.__new__(CreativeWritingEval)
    ev.config = {'5_scale': five_scale}
    rows = [{'raw_output': r} for r in raws]
    return [r['cleaned_output'] for r in ev.parse_llm_outputs(rows)]


class TestTTCWBinaryVerdict(unittest.TestCase):

    # Abridged from data/output/b1/ttcw/gpt41mini: reasoning first, verdict last.
    YES = ("The story builds a gradual, atmospheric tension around the confusion "
           "experienced by new mothers in the maternity ward.\n\n"
           "- The ending feels earned because it is the moment the narrative has "
           "been building toward.\n\n**Answer**: Yes")
    NO = ("The conclusion feels somewhat arbitrary and lacking a clear payoff.\n\n"
          "**Answer**: No")

    def test_verdict_is_read_from_the_end_not_the_first_20_chars(self):
        # The whole bug in one assertion.
        self.assertEqual(_clean([self.YES, self.NO]), [1, 0])

    def test_first_20_chars_of_a_real_yes_contain_no_verdict(self):
        # Why the old slice could not work: there is nothing to find up there.
        self.assertNotIn('yes', self.YES.strip()[:20].lower())

    def test_reply_without_the_marker_is_invalid_not_a_no(self):
        # Two of 15 qwen replies reasoned to a clear conclusion and never emitted
        # '**Answer**:'. Scoring those 0 makes a format failure look like a vote.
        unmarked = ("Overall, the story does not demonstrate originality or "
                    "inventiveness in its form or structure.")
        self.assertEqual(_clean([unmarked]), [-1])

    def test_yesterday_is_not_a_yes(self):
        # No whole-word verdict here at all, so the honest answer is "invalid",
        # not "no" -- and certainly not the 1 a substring test would have given.
        self.assertEqual(_clean(["**Answer**: Nothing happened yesterday."]), [-1])

    def test_bracketed_verdict_is_read(self):
        # The prompt literally shows '**Answer**: [Yes/No]', so models bracket it.
        self.assertEqual(_clean(["**Answer**: [Yes]", "**Answer**: [No]"]), [1, 0])

    def test_last_marker_wins_when_the_judge_restates_the_format(self):
        restated = ('I will answer in the form "**Answer**: [Yes/No]".\n'
                    'The ending is not earned.\n**Answer**: No')
        self.assertEqual(_clean([restated]), [0])


class TestTTCWScoringRule(unittest.TestCase):
    """An unreadable verdict is recorded as -1 but scores as 0.

    Joey's rule, 2026-09-04: "可以标记为 -1，但是最后计算分数的时候当作
    '没有回答正确'，因为格式不正确也是不正确". Both halves are load-bearing.
    Keeping -1 in the record is what lets a broken judge be told apart from a
    harsh one. Scoring it 0 rather than dropping the row is what keeps the
    denominator honest -- excluding unreadable answers would let a judge that
    fails the format raise the reported pass rate.
    """

    def _report(self, raws):
        ev = CreativeWritingEval.__new__(CreativeWritingEval)
        ev.config = {}
        rows = ev.parse_llm_outputs([{"raw_output": r} for r in raws])
        for i, row in enumerate(rows):
            row["prompt_id"] = ["story0", 1]
        return ev.generate_eval_report(rows), rows

    def test_minus_one_is_recorded_but_scores_zero(self):
        rows = CreativeWritingEval.__new__(CreativeWritingEval)
        rows.config = {}
        cleaned = rows.parse_llm_outputs([{"raw_output": "no marker at all"}])
        self.assertEqual(cleaned[0]["cleaned_output"], -1)   # the record keeps it
        cleaned[0]["prompt_id"] = ["story0", 1]
        report = rows.generate_eval_report(cleaned)
        cell = [v for k, v in report.iloc[0].items()
                if k not in ("dp_id", "unparseable_verdicts")]
        self.assertEqual(cell, [0])                          # the score does not

    def test_unparseable_count_is_reported(self):
        ev = CreativeWritingEval.__new__(CreativeWritingEval)
        ev.config = {}
        cleaned = ev.parse_llm_outputs([{"raw_output": "no marker"}])
        cleaned[0]["prompt_id"] = ["story0", 1]
        report = ev.generate_eval_report(cleaned)
        self.assertEqual(int(report.iloc[0]["unparseable_verdicts"]), 1)

    def test_a_real_no_reports_zero_unparseable(self):
        ev = CreativeWritingEval.__new__(CreativeWritingEval)
        ev.config = {}
        cleaned = ev.parse_llm_outputs([{"raw_output": "**Answer**: No"}])
        cleaned[0]["prompt_id"] = ["story0", 1]
        report = ev.generate_eval_report(cleaned)
        self.assertEqual(int(report.iloc[0]["unparseable_verdicts"]), 0)


class TestTTCWFiveScaleVerdict(unittest.TestCase):

    def test_plain_digit(self):
        self.assertEqual(_clean(["**Answer**: 4"], five_scale=True), [4])

    def test_bracketed_score(self):
        # '**Answer**: [4]' -- the shape the prompt asks for. int('[') raises, so
        # this used to score -1: a judge that answered correctly, recorded as
        # having failed to answer.
        self.assertEqual(_clean(["**Answer**: [4]"], five_scale=True), [4])

    def test_unanswerable_is_minus_one(self):
        self.assertEqual(_clean(["I cannot rate this."], five_scale=True), [-1])


class TestCreativeMathVerdict(unittest.TestCase):

    def test_whole_word_only(self):
        self.assertEqual(extract_yes_no("YES, the solution is valid."), "YES")
        self.assertEqual(extract_yes_no("No. It contradicts step 3."), "NO")

    def test_api_error_is_not_a_verdict(self):
        self.assertIsNone(extract_yes_no("API Error: rate limit exceeded"))

    def test_missing_verdict_is_none_not_no(self):
        self.assertIsNone(extract_yes_no("The reasoning is hard to follow."))


if __name__ == "__main__":
    unittest.main()
