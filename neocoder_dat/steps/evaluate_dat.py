import re
import json
import click
from src.utils.dat_score import Model

@click.command()
@click.option("--result-path", type=str)

def main(result_path):
    model = Model("/afs/crc.nd.edu/group/dmsquare/vol4/ylu33/projects/creativity/glove.840B.300d.txt", "/afs/crc.nd.edu/user/y/ylu33/Private/workspace/NeoCoder/words.txt")
    dat_inference_result_path = result_path

    with open(dat_inference_result_path, "r") as f:
        data = json.load(f)

    text = """## Okay, here are 10 single-word nouns chosen for maximum irrelevance across their meanings and typical contexts:\n\n1.  **Quark** (a fundamental particle in physics)\n2.  **Horizon** (the line where the earth's surface and the sky appear to meet)\n3.  **Syllable** (a unit of pronunciation having one vowel sound)\n4.  **Gasket** (a shaped piece or ring of rubber or other material sealing the junction between two surfaces)\n5.  **Nostalgia** (a sentimental longing or wistful affection for a period in the past)\n6.  **Fungus** (any of a group of spore-producing organisms feeding on organic matter, e.g., molds, yeast, mushrooms)\n7.  **Kilogram** (the SI base unit of mass)\n8.  **Tundra** (a vast, flat, treeless Arctic region in which the subsoil is permanently frozen)\n9.  **Ambiguity** (the quality of being open to more than one interpretation; inexactness)\n10. **Waltz** (a dance in triple time performed by a couple)"""

    def parse_first_seven(item):
        text = item['output'].strip()

        # remove string before the first colon
        text = re.sub(r'^[^:]*:', '', text)

        # remove word "user" and "system" in the text
        text = re.sub(r'user', '', text)
        text = re.sub(r'system', '', text)

        # Remove sequences of underscores (10 or more consecutive underscores)
        text = re.sub(r'_{5,}', '', text)

        # Check if "Your answer:" is present. If yes, extract only the text after it until the newline.
        answer_match = re.search(r'Your answer:\s*(.*?)(?:[\n\r]|$)', text)
        if answer_match:
            text = answer_match.group(1)

        # if **WORD** is present, extract all WORD from it
        word_matches = re.findall(r'\*\*(.*?)\*\*', text)
        if word_matches:
            text = ' '.join(word_matches)
        else:
            # If no **WORD** is found, use the entire text
            text = re.sub(r'\*\*', '', text)

        # Split the text by any sequence of digits or non-word characters.
        tokens = re.split(r'[\d\W]+', text)

        words = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            # Ignore tokens that are not a single word (if they contain whitespace)
            if ' ' in token:
                continue
            words.append(token)
            if len(words) == 10:
                break
        return words

    results = []
    for item in data:
        words = parse_first_seven(item)

        score = model.dat(words)

        results.append({
            "words": words,
            "score": score
        })
    # Save file path by replacing inference in the result-path by evaluation
    dat_save_path = dat_inference_result_path.replace("inference", "evaluation")

    with open(dat_save_path, "w") as f:
        json.dump(results, f, indent=4)

    # find average score
    total = 0
    count = 0
    for item in results:
        if item['score'] is not None:
            total += item['score']
            count += 1
    average = total / count
    print("Average DAT score:", average)

if __name__ == "__main__":
    main()