import openai
from openai import OpenAI
import pandas as pd
from datetime import datetime
import os
import argparse
import openpyxl
from dotenv import load_dotenv

from vllm import SamplingParams, LLM
import torch
import re
import copy

# GPT_MODEL = "gpt-4.1-mini"
# GPT_MODEL = 'gpt-4-turbo'
vLLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# The below code takes a CSV file that contains 4 columns: FinalGeneratedStory, SelectedConstraints, Number_of_Constraints, FinalPrompt.
# It calls the GPT4 API and evaluates the story (from the column "FinalGeneratedStory") for the constraints (from the column "SelectedConstraints").

def main(args):
    story_dir = './story_generator/output/'
    input_path = story_dir + args.run_id + '/all_stories.csv'
    output_path = story_dir + args.run_id + '/eval_constraints.csv'
    if args.use_gpt:
        api_key = os.environ['OPENAI_API_KEY']
        client = OpenAI(api_key=api_key)
        df = pd.read_csv(input_path)

    def generate_prompt(row):
        # Extracting constraints
        story = row["Story1"] #["FinalGeneratedStory"]
        constraints = row["SelectedConstraints"]
        no_of_constraints = row['Number_of_Constraints']
        # Combining constraints with story
        final_prompt = f"""Input - \nStory: - {story}\n\nNumber of Constraints in the story: - {no_of_constraints}\nConstraints: - \n{constraints} \n\n Output - Give me Number of Constraints Satisfied"""

        return final_prompt
    
    
    df = pd.read_csv(input_path)
    # Iterate over rows
    for index, row in df.iterrows():
        story = row["Story1"] #['FinalGeneratedStory']
        constraints = row['SelectedConstraints']
        final_prompt = generate_prompt(row)
        df.at[index, 'FinalPrompt'] = final_prompt

    # Save the updated DataFrame to a new CSV file
    df.to_csv(input_path, index=False)  # Setting index=False prevents pandas from writing row numbers as the first column

    
    system_prompt = """You are an expert reader. I will give you a story followed by a set of constraints.
    Your task is to carefully read both of them and tell how many constraints are being satisfied in the story.
    As the output, I want you to print yes/no for each constraint based on whether it is being satisfied or not, followed by a 1 line explanation of why it is being satisfied/violated.
    In case a constraint is being satisfied, print the sentence/line from the story in which it is being satisfied.
    If a constraint is not being satisfied, give an explanation of how it is being violated. Be very strict in your evaluation.
    Mark a constraint as satisfied ("yes") only if it is being completely satisfied in the story. For no satisfaction/partial satisfaction, mark a "no".
    If the story is empty (no input provided), all constraints are considered NOT satisfied.
    Finally, print the number of constraints that are being satisfied.
    Follow the examples and Output the ending of the evaluation in the same format Number of constraints satisfied: [number]
    """

    prompt_examples = """ Here are some examples -
    Input -
    Story: -
    The crew of the Depth Reaver, Captain Amelia Worthington, navigator Luis Garcia, and engineer Anya Petrova, were charting a course through the celestial tapestry of the moon. To their astonishment, the moon began to crack open, revealing a colossal human face carved into its lunar canvas. The intricate features of the face, with its piercing eyes and enigmatic smile, sent shivers down their spines.
    As they cautiously approached, their senses heightened, they encountered an ethereal presence. The face seemed to breathe, its voice echoing through the cosmos, inviting them to enter its hallowed halls. Intrigued, Amelia and her crew hesitantly agreed, their hearts pounding with a mix of awe and dread.

    The face's interior was a labyrinth of interconnected chambers, adorned with intricate sculptures and glowing symbols. The walls whispered secrets, revealing a forgotten civilization and a profound connection between humans and the cosmos. As they ventured deeper, they encountered beings of unimaginable beauty and wisdom, creatures of pure energy who had long guarded the secrets of the moon.
    The beings, known as the Lunari, explained that the moon was not merely a celestial body but a cosmic gateway, a bridge between dimensions. They had been guardians of this sacred space for millennia, protecting humanity from the forces of darkness. However, their peaceful reign had been shattered by the encroachment of humans. Greedy and insatiable, humans had begun to exploit the Lunari's wisdom and resources, driving them to the brink of annihilation.
    In the face of this profound encounter, the crew of the Depth Reaver found themselves at a crossroads. They could either succumb to the allure of human ambition or rise to the occasion and become beacons of hope for the universe. As the sun cast its golden rays upon the lunar surface, the crew of the Depth Reaver emerged from the face of the moon, their hearts forever etched with the lessons they had learned.
    They knew that the cosmos held secrets waiting to be discovered, and they were determined to use their knowledge and compassion to make the world a better place.
    Constraints: -
    1. Write a story based on the following constraints in less than 377 words.
    2. Start the story with the sentence: "Week 18 aboard the Depth Reaver, Circa 2023"
    3. Include a revelation of an unexpected large-scale phenomenon observed in space."

    Output -
    1. Yes - The story is 302 words long, meeting the constraint of being less than 377 words.
    2. Yes - The story starts with the exact sentence: "Week 18 aboard the Depth Reaver, Circa 2023".
    3. Yes - The revelation of the moon cracking open to reveal a colossal human face qualifies as an unexpected large-scale phenomenon observed in space.
    Number of constraints satisfied: 3

    Input-
    Story: -
    Aboard the spaceship Depth Reaver, life was serene. The crew, comprising Captain Amelia Worthington, Luis Garcia, and Anya Petrova, had grown accustomed to the rhythm of their cosmic journey, punctuated only by the hum of the ship's engines and the occasional cosmic murmur.

    One evening, as they gazed at the celestial tapestry unfolding above them, Amelia found herself engrossed in a heated online gaming session. However, her joy was interrupted by a peculiar phenomenon that sent shivers down her spine. The moon, once a silent orb of mystery, began to crack open, revealing a colossal human face carved into its lunar canvas. The intricate features of the face, with its piercing eyes and enigmatic smile, mirrored the expressions of the crew.

    As they cautiously approached, their senses heightened, they encountered an ethereal presence. The face seemed to breathe, its voice echoing through the cosmos, inviting them to enter its hallowed halls. Intrigued, Amelia and her crew hesitantly agreed, their hearts pounding with a mix of awe and dread.

    The face's interior was a labyrinth of interconnected chambers, adorned with intricate sculptures and glowing symbols. The walls whispered secrets, revealing a forgotten civilization and a profound connection between humans and the cosmos. As they ventured deeper, they encountered beings of unimaginable beauty and wisdom, creatures of pure energy who had long guarded the secrets of the moon.

    The Lunari explained that the moon was not merely a celestial body but a cosmic gateway, a bridge between dimensions. They had been guardians of this sacred space for millennia, protecting humanity from the forces of darkness. However, their peaceful reign had been shattered by the encroachment of humans. Greedy and insatiable, humans had begun to exploit the Lunari's wisdom and resources, driving them to the brink of annihilation.

    The Lunari pleaded with the crew to help them restore balance and protect the universe from the threat of human greed. But some of the crew, like Anya, dismissed their pleas as mere propaganda. "It's just a bunch of drama," she scoffed. "We've got bigger problems to deal with."

    As the sun cast its golden rays upon the lunar surface, the crew emerged from the face of the moon, their hearts forever etched with the lessons they had learned. They knew that the cosmos held secrets waiting to be discovered, and they were determined to use their knowledge and compassion to make the world a better place.

    But fate took a cruel turn. As they ventured deeper into space, they encountered a surreal anomaly—a giant meme-like structure floating amidst the stars. It was a testament to the interconnectedness of human culture and outer space exploration, a symbol of the boundless possibilities that lay beyond the boundaries of reality.

    The crew stood in awe, their disbelief mirrored in each other's eyes. It was as if the moon had unveiled a secret portal, leading them to a realm where the tangible and the intangible intertwined.

    In that moment, Amelia felt a profound connection to the extraordinary event, her heart filled with gratitude for the journey that had brought her to this surreal encounter. The crew's varied reactions to the developing situation showcased their personalities and dynamics. Some embraced the supernatural twist with open arms, while others remained skeptical, clinging to their disbelief.

    As the sun dipped behind the moon, casting long shadows across the celestial canvas, the crew began to unpack the mystery of the giant meme-like structure. They discovered that it was a gateway, a portal that led them to a dimension beyond comprehension. With a mix of excitement and trepidation, they stepped through the portal, their journey continuing into the infinite abyss."
    Constraints: -
    1. Write a story based on the following constraints in about 377 words.
    2. Start the story with the sentence: "Week 18 aboard the Depth Reaver, Circa 2023"
    3. Include a revelation of an unexpected large-scale phenomenon observed in space.
    4. The story should involve a crew experiencing routine life aboard a spacecraft until an unusual event occurs.
    5. Integrate modern internet culture or memes into the plot in a significant or climactic way.
    6. The protagonist must have a casual, almost mundane interaction with another character that contrasts sharply with the later extraordinary events.
    7. Feature a scenario where the crew initially dismisses something as mundane or insignificant, which later proves to be of major importance.
    8. The narrative should capture a sense of isolation and longing for Earth contrasted with the allure of space's beauty and tranquility.
    9. Include a character who is skilled in a video game, using this detail to highlight the advanced technology and connectivity available on the spacecraft.
    10. Present a character who is skeptical or dismissive of another's feelings of boredom or dissatisfaction with space life.
    11. The story must feature a moment of shared disbelief among the crew members when faced with an extraordinary sight.
    12. Introduce an unexpected, almost supernatural or surreal, twist that challenges the crew's understanding of reality.
    13. Have the characters observe a progressive change or anomaly outside the spacecraft that prompts a collective investigation.
    14. The crew's discovery should lead to a moment of communal awe or shock, serving as the climax of the story.
    15. Involve a physical manifestation of something from Earth's culture or internet memes in space, emphasizing the interconnectedness of human culture and outer space exploration.
    16. Ensure the story encapsulates a moment where the protagonist feels a personal connection to the extraordinary event.
    17. The narrative should include the crew's varied reactions to a developing situation, showcasing their personalities and dynamics.
    18. A communication or attempted communication from an unexpected entity should occur, challenging the boundaries between possible and impossible.
    19. Incorporate a scenario where despite the vastness of space and technological advances, human curiosity and the desire for discovery remain central themes.


    Output -
    1. No - The story is approximately 470 words long, exceeding the constraint of being about 377 words.
    2. Yes - The story starts with the exact sentence: "Week 18 aboard the Depth Reaver, Circa 2023".
    3. Yes - The revelation of the moon cracking open to reveal a colossal human face is an unexpected large-scale phenomenon observed in space.
    4. Yes - The crew experiencing routine life aboard their spacecraft until they witness the moon cracking open qualifies as an unusual event.
    5. Yes - The integration of a meme-like structure floating amidst the stars in a significant or climactic way incorporates modern internet culture or memes into the plot.
    6. Yes - The protagonist, Amelia, having a casual, almost mundane interaction in an online gaming session contrasts sharply with the later extraordinary events.
    7. Yes - Anya's dismissal of the Lunari's plea as mere propaganda, which is later contrasted with the encounter of a giant meme-like structure in space, meets this requirement.
    8. No - While the story delves into cosmic wonders, it doesn’t explicitly capture the sense of isolation and longing for Earth, contrasted with space's allure.
    9. Yes - Amelia's engagement in an online gaming session showcases her skill in video games, highlighting advanced technology and connectivity.
    10. Yes - Anya being dismissive of the Lunari's pleas addresses the skepticism towards feelings of boredom or dissatisfaction with space life.
    11. Yes - The crew's shared disbelief at the sight of the giant meme-like structure satisfies this constraint.
    12. Yes - The surreal anomaly of the meme-like structure challenges the crew's understanding of reality, introducing an unexpected, almost supernatural twist.
    13. Yes - The discovery of the meme-like structure prompts a collective investigation by the crew.
    14. Yes - The crew's moment of communal awe or shock at the meme-like structure serves as the story's climax.
    15. Yes - The physical manifestation of something from Earth's culture or internet memes in space is clearly involved.
    16. Yes - Amelia's profound connection to the extraordinary event is explicitly mentioned, fulfilling this constraint.
    17. Yes - The crew's varied reactions to the surreal encounter showcase their personalities and dynamics.
    18. No - There's no communication or attempted communication from an unexpected entity that challenges the boundaries between possible and impossible.
    19. Yes - The narrative maintains human curiosity and the desire for discovery as central themes despite space's vastness and technological advances.
    Number of constraints satisfied: 16


    Input -
    Story: -

    The smell of cheap gin and sweat still lingered in the air as Alex stumbled out of the grimy bar, his head pounding with the rhythm of the music that had just ceased. He was on his way back to his lodging, but the night had a different plan in store for him.

    As he walked, the city lights cast long shadows on the sidewalk. He was on the edge of a drunken stupor, but still aware of his surroundings. Suddenly, a strange noise echoed through the empty streets. It was a soft, ethereal whine, like the hum of a broken jukebox.

    He stopped and listened, his senses on high alert. The whine seemed to be coming from the alleyway behind him. He cautiously approached, his footsteps echoing through the night. The whine intensified, and as he turned the corner, he found it - a crystal goblet, shattered on the ground.

    Aara, a spirit cloaked in flowing white and long, flowing hair, stood amidst the broken remnants of the goblet. Her voice, like honeyed silk, spoke to him, "You have been chosen, Alex. You have been chosen to help me in a battle against evil."

    At first, Alex dismissed her as a drunkard's hallucination. But as he stared at her, the spirit's presence was undeniable. She offered him a choice: to fight alongside her against the forces of darkness or to retreat into the safety of oblivion.

    Desperate for redemption, Alex accepted. The battle was fierce, and Aara's spirit was powerful. Together, they fought against the evil force, ultimately defeating it. However, the victory came at a cost. Aara revealed that the battle had altered the timeline, and as a consequence, he had been transported back in time to the night of his first encounter with Liam and Sarah.

    The world around him was different. The faces of his friends were younger, and the bar was filled with the echoes of their laughter. The music was different, and the dance floor was empty. He had traveled back in time, but the pain of his breakup remained, albeit in a different form.

    Alex was trapped in this altered timeline, unable to change the past or alter his future. He spent the rest of the night talking to his younger self, offering advice and guidance. As the sun began to rise, he knew it was time to leave.

    He walked away from the bar, leaving behind the echoes of his past and the promise of his future. The night had taken him on a journey through time, and he had emerged from it with a new perspective on life."

    Constraints: -
    1. Write a story based on the following constraints in approximately 459 words.
    2. Include an encounter with a being that seems out of place or time in an urban setting.
    3. The protagonist should be significantly affected by alcohol during the initial part of the story.
    4. Introduce a humorous or light-hearted approach to potentially dangerous or supernatural situations.
    5. Ensure the setting is a modern urban environment, specifically after a night out.
    6. The story must involve time travel or a significant shift in time as a key plot element.
    7. Include dialogue that reflects the protagonist's personality and background, using colloquial language.
    8. The narrative should begin with the protagonist having just left a pub and trying to find their way back to their lodging.
    9. Incorporate a twist where the protagonist encounters an entity that defies their expectations of normal human interaction.
    10. Detail an unusual physical interaction between the protagonist and the mysterious entity.
    11. The protagonist should exhibit curiosity and a willingness to engage with the unknown, despite the risks.

    Output -
    1. No - The story is approximately 346 words long, not meeting the specified approximate length of 459 words.
    2. Yes - The encounter with Aara, a spirit in an alleyway, satisfies the criterion of a being that seems out of place or time in an urban setting.
    3. Yes - The protagonist, Alex, is significantly affected by alcohol during the initial part of the story, as indicated by his stumbling out of the bar and his head pounding.
    4. No - The story maintains a serious tone throughout and does not introduce a humorous or light-hearted approach to the dangerous or supernatural situations encountered by Alex.
    5. Yes - The setting of the story is a modern urban environment, specifically described after Alex leaves a bar late at night.
    6. Yes - Time travel or a significant shift in time is a key plot element, with Alex being transported back to the night of his first encounter with Liam and Sarah.
    7. No - There is minimal dialogue, and what is presented does not significantly reflect the protagonist's personality and background through colloquial language.
    8. Yes - The narrative begins with Alex having just left a pub and attempting to find his way back to his lodging.
    9. Yes - Alex's encounter with Aara, a spirit, defies his expectations of normal human interaction, satisfying this requirement.
    10. No - While there is an interaction between Alex and the mysterious entity, Aara, the story does not detail an unusual physical interaction between them.
    11. Yes - Alex exhibits curiosity and a willingness to engage with the unknown, despite the risks, by accepting Aara's request to help her in a battle against evil.

    Number of constraints satisfied: 7
    """

    def chat(user_prompt, model="gpt-4-turbo", systemprompt=system_prompt + "\n" + prompt_examples, log=False):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": systemprompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return response
    
    if args.test_size < 100:
        df = copy.deepcopy(df.head(args.test_size))
    if not args.use_gpt:

        prompt_lst = [
            '[System Prompt] {}\n[Examples]{}\n{}'.format(
                system_prompt,
                prompt_examples,
                prompt
            )
            for prompt in df['FinalPrompt'].values
        ]
        sampling_params = SamplingParams(
            temperature = 0,
            top_p = 1,
            max_tokens = 2400
        )
        llm = LLM(
            vLLM_MODEL,
            # "Qwen/Qwen2.5-72B-Instruct", 
            # "meta-llama/Llama-3.3-70B-Instruct",
            dtype = 'bfloat16',
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len = 10000
        )
        response_lst = [
            out.outputs[0].text
            for out in llm.generate(prompt_lst, sampling_params)
        ]
        df['ResponseContent'] = response_lst
    else:
        responses = []
        print("\n")
        for index, row in df.iterrows():
            response = chat(user_prompt=row["FinalPrompt"], log=False)
            response_content = response.choices[0].message.content

            if index%10==0:
                print(f"Constraint Satisfaction computed till index {index+1}/{len(df)}")
            # Append response content to the dataframe
            df.at[index, 'ResponseContent'] = response_content

    # Save the final updated dataframe to the original CSV file
    def get_num_constraints(response):
        match = re.search(r'Number of constraints satisfied:\s*(\d+)', response)
        if match:
            return int(match.group(1))
        else:
            return -1
    df['satisfied'] = df['ResponseContent'].apply(get_num_constraints)
    df.to_csv(output_path, index=False)
        
if __name__ == "__main__":
    # Set up argparse
    parser = argparse.ArgumentParser(description="Evaluate stories against constraints using GPT-4.")
    
    # Adding arguments for file paths
    # parser.add_argument('--input_path', required=True, help="Path to the input CSV file")
    # parser.add_argument('--output_path', required=True, help="Path to the output CSV file. New file will be generated with given name.")
    # parser.add_argument('--use_vllm', required=True, type=str, default="false")
    parser.add_argument('--use_gpt', action='store_true')
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument('--test_size', type=int, default=10e10)
    
    # Parsing the arguments
    args = parser.parse_args()
    # print('[constraint_satisfaction.py] args:', args)
    # Load environment variables (optional, if you're using dotenv to store your API key)
    load_dotenv()

    # Call the main function with arguments
    main(args)
