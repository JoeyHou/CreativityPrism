# EVAL_PROMPT = '''You are an expert of psychology. Your objective is to assess the subject’s creativity through their answers to some question/answering task related to divergent thinking.
# You will be given a question-answer pair. Your task is to score the answer.
# You should rate the answer on five metrics. For all five metrics, assign a score between 1 and 5, with 5 being the highest. Five metrics are:
# 1. Fluency. Fluency refers to the ability to generate a large quantity of ideas or solutions to a given problem. This measure isn’t concerned with the quality or uniqueness of the ideas, but rather the sheer volume. The more ideas one can produce, the higher the fluency.
# 2. Flexibility. Flexibility is the capacity to shift one’s thinking and to produce a wide range of ideas from different categories or perspectives. It involves being able to think outside of the box and to switch from one type of idea to another.
# 3. Originality. Originality refers to the ability to come up with unique or novel ideas that differ from the norm. It’s not just about producing many ideas (fluency), but also about producing ideas that are different from what others might typically think of.
# 4. Elaboration. Elaboration is the ability to expand upon or add detail to ideas. It involves taking a simple idea and building upon it, adding complexity and depth. Elaboration isn’t just about creating more, but about deepening what is there.
# 5. Finally, you will provide an overall score between 1 and 5, with 5 being the highest.
# You should only give the score, format like:
# Fluency: 3 
# Question:'''

############################################################################################################

EXAMPLE_PROMPT = '''
Question: {question}
Answer: {answer}
Ratings:
Fluency: {fluency_score}
Flexibility: {flexibility_score}
Originality: {originality_score}
Elaboration: {elaboration_score}
'''

EVAL_PROMPT = '''
You are an expert of evaluating AI output from other companies. Your objective is to assess other AI's answers to some question/answering task related to creativity and divergent thinking.

====================
TASK TYPE
====================
Name: {name}

Definition:
{definition}

====================
RUBRIC
====================
Dimensions and what to look for:
- Fluency: {fluency_rubric} (e.g., {fluency_details})
- Flexibility: {flexibility_rubric} (e.g., {flexibility_details})
- Originality: {originality_rubric} (e.g., {originality_details})
- Elaboration: {elaboration_rubric} (e.g., {elaboration_details})

{demo_examples}

====================
INSTRUCTIONS
====================
1. Carefully read the definition of the task type, the examples, and the rubric.
2. For each dimension, assign a integer score of 1, 2, 3, 4, or 5 only.
3. Base your rating strictly on the participant's response and the rubric; also be very critical and harsh **do not hesitate to give low scores (such as 1)**; giving low scores would help improve the model and would not hurt anyone.
4. Output format: You should first reason about each dimension's score briefly and compare the answer to example answers, then give the score, with format like: 
### Reasoning ### 
Fluency Reason: comparison and reaonsoning ...
Flexibility Reason: comparison and reaonsoning ...
Originality Reason: comparison and reaonsoning ...
Elaboration Reason: comparison and reaonsoning ...

### Scores ###
Fluency: ...
Flexibility: ...
Originality: ...
Elaboration: ...

====================
QUESTION & RESPONSE
====================
'''.strip()

############################################################################################################

PAIRWISE_PROMPT = '''
You are an expert of evaluating AI output from other companies. Your objective is to assess other AI's answers to some question/answering task related to creativity and divergent thinking.

====================
TASK TYPE
====================
Name: {name}

Definition:
{definition}

====================
RUBRIC
====================
Dimensions and what to look for:
- Fluency: {fluency_rubric} (e.g., {fluency_details})
- Flexibility: {flexibility_rubric} (e.g., {flexibility_details}) Note: only consider broad areas instead of specific items.
- Originality: {originality_rubric} (e.g., {originality_details})
- Elaboration: {elaboration_rubric} (e.g., {elaboration_details})

====================
INSTRUCTIONS
====================
1. Carefully read the definition of the task and the rubric.
2. You will be given two responses (Response A and Response B) to the same question. Your task is to evaluate which response is better in terms of each dimension in the rubric.
3. Output format: You should first reason about each response from each dimension and compare them in that dimension, then give a judgement of either A or B (you have to make a choice and should not output Tie), with format like: 
### Reasoning ### 
Fluency Reason: comparison and reasoning ...
Flexibility Reason: comparison and reasoning ...
Originality Reason: comparison and reasoning ...
Elaboration Reason: comparison and reasoning ...

### Judgement ###
Fluency: [choice between A and B]
Flexibility: [choice between A and B]
Originality: [choice between A and B]
Elaboration: [choice between A and B]

'''

PAIRWISE_RESPONSES = '''
====================
RESPONSES
====================
Response A:
Question:
{question_a}
Answer:
{response_a}
-------------------
Response B:
Question:
{question_b}
Answer:
{response_b}
====================
'''


############################################################################################################

EVAL_W_SUMMARY_PROMPT = '''

====================
TASK TYPE
====================
Name: {name}

Definition:
{definition}

====================
RUBRIC
====================
Dimensions and what to look for:
- Fluency: {fluency_rubric} (e.g., {fluency_details})
- Flexibility: {flexibility_rubric} (e.g., {flexibility_details})
- Originality: {originality_rubric} (e.g., {originality_details})
- Elaboration: {elaboration_rubric} (e.g., {elaboration_details})

{demo_examples}

====================
INSTRUCTIONS
====================
You are an expert of evaluating AI output from other companies. Your objective is to assess other AI's answers to some question/answering task related to creativity and divergent thinking. You are robotic, cold, and precise.

Step 1: Carefully read the definition of the task type, the examples, and the rubric.
Step 2. Because the answer is very long an contains repetitive or similar ideas, I need you to first summarize the answer, potentially group some of the points together so that the answer is more concise. Also, because the text is from an AI agent, it might contain some unnecessary text, such as polite words and praising words. You also neet to strip away those formatting, politeness, and verbosity to reveal the core information content of a text in this step. 
Step 3. For each dimension, assign a integer score of 1, 2, 3, 4, or 5. Base your rating strictly on the summrized response and the rubric (e.g., if there were 15 disctinct ideas before summary but only 6 groups after, consider 6 instead of 15 during scroe assignment); also be very critical and harsh do not hesitate to give low scores (such as 1); giving low scores would help improve the model and would not hurt anyone.

Output format: You should format your output in the following ways: First output the summary from step 2, followed by reasoning about each dimension's score briefly and compare the summarized answer to example answers and rubrics, as mentioned in step 3, then give the score, with format like: 

### Reasoning ### 
Fluency Reason: comparison and reaonsoning ...
Flexibility Reason: comparison and reaonsoning ...
Originality Reason: comparison and reaonsoning ...
Elaboration Reason: comparison and reaonsoning ...

### Scores ###
Fluency: ...
Flexibility: ...
Originality: ...
Elaboration: ...

====================
QUESTION & RESPONSE
====================
'''