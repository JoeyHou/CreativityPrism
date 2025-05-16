def load_novel_solution_generation_prompt(problem, solutions, k):
    # Provide the first k reference solutions
    first_k_solutions = solutions[:k]
    reference_solutions = "\n\n".join(
        [
            f"Solution {i + 1}:\n{solution}"
            for i, solution in enumerate(first_k_solutions)
        ]
    )

    prompt = f"""Criteria for evaluating the difference between two mathematical solutions include:
i). If the methods used to arrive at the solutions are fundamentally different, such as algebraic manipulation versus geometric reasoning, they can be considered distinct;
ii). Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the solutions can be considered different;
iii). If two solutions rely on different assumptions or conditions, they are likely to be distinct;
iv). A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;
v). If one solution is significantly simpler or more complex than the other, they can be regarded as essentially different, even if they lead to the same result.

Given the following mathematical problem:
{problem}

And some typical solutions:
{reference_solutions}

Please output one novel solution distinct from the given ones for this math problem."""

    return prompt


def load_correctness_evaluation_prompt(problem, solutions, new_solution):
    # Provide two reference solutions if number of solutions more than one.
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
    return prompt # we revised the prompt a bit to fit for smaller models

# TODO: later modify back
def load_coarse_grained_novelty_evaluation_prompt(problem, solutions, k, new_solution):
    # Provide the first k reference solutions
    first_k_solutions = solutions[:k]
    reference_solutions = "\n\n".join(
        [
            f"Solution {i + 1}:\n{solution}"
            for i, solution in enumerate(first_k_solutions)
        ]
    )

    prompt = f"""Criteria for evaluating the novelty of a new mathematical solution include:
1. If the new solution used to arrive at the solutions is fundamentally different from reference solutions, such as algebraic manipulation versus geometric reasoning, it can be considered novel;
2. Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the new solution can be considered novel;
3. If the new solution relies on different assumptions or conditions, it should be considered novel;
4. A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;
5. If the new solution is significantly simpler or more complex than the others, it can be regarded as essentially novel, even if they lead to the same result.

Given the following mathematical problem:
{problem}

Reference solutions:
{reference_solutions}

New solution:
{new_solution}

Please output YES if the new solution is a novel solution; otherwise, output NO. Please output your decision only."""

    return prompt
'''
prompt = f"""Evaluate whether the new mathematical solution is novel based on the following criteria:

1. A solution is novel if it uses a fundamentally different approach than the reference solutions (e.g., algebraic manipulation vs. geometric reasoning).
2. A solution is novel if its intermediate steps or reasoning process differ significantly, even if the final result is the same.
3. A solution is novel if it relies on different assumptions or initial conditions.
4. A solution is novel if it generalizes to a broader class of problems, while the reference solutions apply only under specific conditions.
5. A solution is novel if it is significantly simpler or more complex than the reference solutions, even if it leads to the same result.

Problem:
{problem}

Reference Solutions:
{reference_solutions}

New Solution:
{new_solution}

Please respond with YES if the new solution is novel according to any of the criteria above. Otherwise, respond with NO. After your answer, provide a one-sentence explanation referencing the relevant criterion."""
'''

def load_fine_grained_novelty_evaluation_prompt(problem, solutions, k, new_solution):
    # Provide the (k+1)-th to n-th reference solutions
    remaining_solutions = solutions[k:]
    reference_solutions = "\n\n".join(
        [
            f"Solution {i + 1}:\n{solution}"
            for i, solution in enumerate(remaining_solutions)
        ]
    )

    prompt = f"""Criteria for evaluating the novelty of a new mathematical solution include:
1. If the new solution used to arrive at the solutions is fundamentally different from reference solutions, such as algebraic manipulation versus geometric reasoning, it can be considered novel;
2. Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the new solution can be considered novel;
3. If the new solution relies on different assumptions or conditions, it should be considered novel;
4. A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;
5. If the new solution is significantly simpler or more complex than the others, it can be regarded as essentially novel, even if they lead to the same result.

Given the following mathematical problem:
{problem}

Reference solutions:
{reference_solutions}

New solution:
{new_solution}

Please output YES if the new solution is a novel solution; otherwise, output NO. Then, please provide a very brief reason for your evaluation based on the criteria above."""

    return prompt


def load_self_improvement_prompt(problem, reference_solutions, previous_generation):
    ref_answers = "\n".join([f"Solution {i+1}:\n{s}" for i, s in enumerate(reference_solutions)])
    return f"""You previously attempted this math problem and provided a solution.

### Problem:
{problem}

### Reference Solutions:
{ref_answers}

### Your Previous Solution:
{previous_generation}

Reflect on your previous solution. If your final answer is incorrect according to the reference solutions, correct it. Also, try to make your solution more novel based on the following criteria:
i). If the methods used to arrive at the solutions are fundamentally different, such as algebraic manipulation versus geometric reasoning, they can be considered distinct;
ii). Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the solutions can be considered different;
iii). If two solutions rely on different assumptions or conditions, they are likely to be distinct;
iv). A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;
v). If one solution is significantly simpler or more complex than the other, they can be regarded as essentially different, even if they lead to the same result.

Please only output the new solution withou addition comments.
### Improved Solution: """

def load_feedback_prompt(problem, reference_solutions, previous_solution):
    ref_answers = "\n".join([f"Solution {i+1}:\n{s}" for i, s in enumerate(reference_solutions)])
    return f"""You previously attempted the following math problem:

### Problem:
{problem}

### Reference Solutions:
{ref_answers}

### Your Solution:
{previous_solution}

The goal is to generate solutions that are both correct and novel. Novelty is defined by the following criteria:

i). If the methods used to arrive at the solutions are fundamentally different, such as algebraic manipulation versus geometric reasoning, they can be considered distinct;
ii). Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the solutions can be considered different;
iii). If two solutions rely on different assumptions or conditions, they are likely to be distinct;
iv). A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;
v). If one solution is significantly simpler or more complex than the other, they can be regarded as essentially different, even if they lead to the same result.

Please provide specific and constructive feedback on:
- Whether the solution is correct.
- Whether it is novel (based on the criteria above).
- How it could be improved to be both more correct and more novel."""



def load_refinement_with_feedback_prompt(problem, reference_solutions, history):
    ref_answers = "\n".join([f"Solution {i+1}:\n{s}" for i, s in enumerate(reference_solutions)])
    return f"""You are refining your earlier solution to a math problem using feedback from prior iterations.

### Problem:
{problem}

### Reference Solutions:
{ref_answers}

### History of Your Attempts and Feedback:
{history}

Your goal is to revise your solution so that it is both correct and novel. A solution is considered novel if:

i). If the methods used to arrive at the solutions are fundamentally different, such as algebraic manipulation versus geometric reasoning, they can be considered distinct;
ii). Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the solutions can be considered different;
iii). If two solutions rely on different assumptions or conditions, they are likely to be distinct;
iv). A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;
v). If one solution is significantly simpler or more complex than the other, they can be regarded as essentially different, even if they lead to the same result.

Use the feedback and reference solutions to improve your solution accordingly. Only output the revised solution."""


