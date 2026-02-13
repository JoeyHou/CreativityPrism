import os
import json
import time
import pandas as pd
from typing import List
from openai import OpenAI

# -----------------------------
# OpenAI client
# -----------------------------
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

MODEL_NAME = "gpt-4.1"

# -----------------------------
# Prompt function (unchanged)
# -----------------------------
def load_correctness_evaluation_prompt(problem, solutions, new_solution):
    if len(solutions) == 1:
        reference_solutions = f"Solution 1:\n{solutions[0]}"
    else:
        reference_solutions = "\n\n".join(
            [
                f"Solution {i + 1}:\n{solution}"
                for i, solution in enumerate(solutions[:2])
            ]
        )

    prompt = f"""Given the following mathematical problem:
{problem}

Reference solutions:
{reference_solutions}

New solution:
{new_solution}

Please output YES if the new solution arrives at the same final result as any of the reference solutions, regardless of whether it uses a novel approach. Output NO otherwise.

For proof-based questions, assess whether the reasoning is logically valid and leads to the correct conclusion.

Then, briefly explain your judgment based on the correctness of the result and reasoning."""
    return prompt


# -----------------------------
# Helper: parse solutions column
# -----------------------------
def parse_solutions(cell) -> List[str]:
    """
    Supports:
    - JSON list string
    - Python list string
    - Single string
    """
    if isinstance(cell, list):
        return cell

    if isinstance(cell, str):
        cell = cell.strip()
        try:
            parsed = json.loads(cell)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        # fallback: treat as single solution
        return [cell]

    raise ValueError(f"Unsupported solutions format: {cell}")


# -----------------------------
# GPT-4.1 judge call
# -----------------------------
def judge_correctness(prompt, temperature=0.0, max_tokens=512):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    text = response.choices[0].message.content.strip()
    return text


# -----------------------------
# Main evaluation loop
# -----------------------------
def main(
    input_csv="/scratch/dkhasha1/bzhang90/creativityprism/v2.0.csv",
    output_csv="/scratch/dkhasha1/bzhang90/creativityprism/v2.0_judged.csv",
    sleep_sec=0.2,
):
    df = pd.read_csv(input_csv)

    judgments = []
    explanations = []

    for idx, row in df.iterrows():
        problem = row["problem"]          # <-- change if needed
        print("problem: ", problem, "\n\n")
        solutions = parse_solutions(row["ground_truth_solutions"])
        assert isinstance(solutions, list), (
            f"`solutions` must be a list, got {type(solutions)} at row {idx}"
        )
        # unwrap accidental stringified list
        if (
            isinstance(solutions, list)
            and len(solutions) == 1
            and isinstance(solutions[0], str)
            and solutions[0].startswith("[")
        ):
            import ast
            solutions = ast.literal_eval(solutions[0])

        assert isinstance(solutions, list)
        assert all(isinstance(s, str) for s in solutions)

        print("provided solutions: ", solutions)

        new_solution = row["cleaned_response"]
        print("model response: ", new_solution)
        print("===========================")
        prompt = load_correctness_evaluation_prompt(
            problem=problem,
            solutions=solutions,
            new_solution=new_solution,
        )
        print(prompt)

        try:
            judge_output = judge_correctness(prompt)
        except Exception as e:
            print(f"[ERROR] Row {idx}: {e}")
            judgments.append("ERROR")
            explanations.append(str(e))
            continue

        # -------------------------
        # Parse YES / NO
        # -------------------------
        first_line = judge_output.splitlines()[0].strip().upper()

        if first_line.startswith("YES"):
            judgment = "YES"
            explanation = judge_output
        elif first_line.startswith("NO"):
            judgment = "NO"
            explanation = judge_output
        else:
            # fallback if model is verbose
            judgment = "YES" if "YES" in judge_output[:20] else "NO"
            explanation = judge_output

        judgments.append(judgment)
        explanations.append(explanation)
        print("The decision is: ", judgment, "Explanation: ", explanation)

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(df)}")

        time.sleep(sleep_sec)

    # -----------------------------
    # Save results
    # -----------------------------
    df["judge_decision"] = judgments
    df["judge_explanation"] = explanations
    df.to_csv(output_csv, index=False)

    print(f"Saved judged file to: {output_csv}")


if __name__ == "__main__":
    main()
