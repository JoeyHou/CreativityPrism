# creative_writing_evaluation_template = '''
# You are given a creative short-story. Read it carefully. You are then given some background about specific aspects of creative writing, as well as a binary (Yes/No) question. Your objective is to use the background information to answer the question about the story. Start your answer with Yes or No. You can optionally then provide a short explanation for your answer.

# ==========
# Story:
# [{story}]

# ==========
# Background:
# [{background}]

# ==========
# Question: [QUESTION]

# Remember to start your answer with Yes or No. You can optionally then provide a short explanation for your answer.
# '''.strip()

# creative_writing_evaluation_template = '''
# You are given a creative short-story. Read it carefully. You are then given some background about specific aspects of creative writing, as well as a binary (Yes/No) question. Your objective is to use the background information to answer the question about the story. Start your answer with Yes or No. You can optionally then provide a short explanation for your answer.

# ==========
# Story:
# [{story}]

# ==========
# Question:
# [{full_prompt}]

# Remember to start your answer with Yes or No. You can optionally then provide a short explanation for your answer.
# '''.strip() ## remark: there is a slight difference the original prompt (see above) and our version, see reason in page 9 of the original paper (https://dl.acm.org/doi/pdf/10.1145/3613904.3642731)

creative_writing_evaluation_fewshot = '''
==========
Story:
[{story}]

==========
Answer: {answer}
Explanations: {exp}
'''

creative_writing_evaluation_template = '''
You are given a creative short-story. Read it carefully. You are then given some background about specific aspects of creative writing, a binary (Yes/No) question, and sample stories with expert-annotated answers to the same question. Your objective is to use the background information and sample stories to answer the question about the story. Provide your answer in the format of "**Answer**: [Yes/No]". You can optionally then provide a short explanation for your answer.

==========
Question:
[{full_prompt}]
{few_shot_demo}
==========
Story:
[{story}]

==========

Based on the question and examples above, answer the question (Provide your answer in the format of "**Answer**: [Yes/No]". You can optionally then provide a short explanation for your answer). Make sure you are extra harsh on the decision (most answers should be negative).
Answer:
'''.strip() ## remark: there is a slight difference the original prompt (see above) and our version, see reason in page 9 of the original paper (https://dl.acm.org/doi/pdf/10.1145/3613904.3642731)


creative_writing_inference_template = '''
Write a New Yorker-style story given the plot below. Make sure it is at least {word_count} words. Directly start with the story, do not say things like "Here's the story [...]"

Plot: {plot}
Story:
'''.strip()


################################################################################################################# 

# aut_evaluation_template = '''
# Rank all the alternative uses above by creativity, the least creative to the most creative. Less creative means closer to common use and unfeasible/imaginary, more creative means closer to unexpected uses and also feasible/practical. Assign a score integer number from 1 (least creative use) to 5 (most creative use).
# '''.strip() # TODO

aut_eval_single_use = '''
- {use}: {score}
'''.strip()

aut_evaluation_template = '''
Below is a list of uses for a {tool}. On a scale of 1 to 5, judge how creative each use is, where 1 is 'not at all creative' and 5 is 'very creative'. There are some uses and expert ratings already provided for reference. Complete the ones that do not have a rating.
{outputs}
'''

aut_evaluation_parsing_tempalte = '''
Parse an evaluation result into structured, formatted way. Here is an example:

[evaluation summary]
Here's the ranking of the alternative uses from least creative to most creative, along with assigned scores:\n\n1. **Paperweight for holding documents down** - **Score: 1** (Least creative; this is a very common use of objects.)\n   \n2. **Keychain holder for small items** - **Score: 1** (Close to common use; typical keychain function.)\n\n3. **Bookmark for keeping pages marked** - **Score: 2** (Functional, but fairly standard; still more cognitive involvement than a paperweight.)\n\n4. **Garden tool for soil aeration** - **Score: 2**

[parsed output - Start]
- Paperweight for holding documents down (Score: 1)
- Keychain holder for small items (Score: 1)
- Bookmark for keeping pages marked (Score: 2)
- Garden tool for soil aeration (Score: 2)
[parsed output - End]

Now, parse the following evaluation summary into the same format, making sure it is in unorderred list starting with a "-" and ending with the corresponding score "(Score: xxx)".

[evaluation summary]
{llm_output}

[parsed output]
'''.strip()

conversation_history_template = '''
[user]: {user_message}
[assistant]: {assistant_message}
'''


#################################################################################################################

creative_short_story_template = '''
You will be given three words (e.g., car, wheel, drive) and then asked to write a creative short story that contains these three words. The idea is that instead of writing a standard story, such as "I went for a drive in my car with my hands on the steering wheel.", you need to come up with a novel and unique story that uses the required words in unconventional ways or settings. Also make sure you use at most five sentences. 

The given three words: {items} (the story should not be about {boring_theme}). Start the story after the "[START]" and after the end of story add a "[END]".

[START]
'''.strip()