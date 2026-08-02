import os
import re
import json
import click
from pathlib import Path
from src.utils.dat_score import Model

TASK_ROOT = Path(__file__).resolve().parent.parent
# ~2 GB, not redistributed with this repo; see the task README for the download step.
DEFAULT_GLOVE_PATH = TASK_ROOT / "embeddings" / "glove" / "glove.840B.300d.txt"
DEFAULT_WORDS_PATH = TASK_ROOT / "words.txt"
GLOVE_DOWNLOAD_URL = "https://nlp.stanford.edu/data/glove.840B.300d.zip"


@click.command()
@click.option("--result-path", type=str)
@click.option("--output-path", type=str, default=None,
              help="Where to write scores. Default: --result-path with 'inference' replaced by 'evaluation'.")
@click.option("--glove-path", type=str, default=None,
              help="GloVe vectors. Falls back to $CREATIVITYPRISM_GLOVE_PATH, then embeddings/glove/glove.840B.300d.txt.")
@click.option("--words-path", type=str, default=None,
              help="Word list. Falls back to $CREATIVITYPRISM_DAT_WORDS, then the bundled words.txt.")

def main(result_path, output_path, glove_path, words_path):
    glove_path = glove_path or os.environ.get("CREATIVITYPRISM_GLOVE_PATH") or str(DEFAULT_GLOVE_PATH)
    words_path = words_path or os.environ.get("CREATIVITYPRISM_DAT_WORDS") or str(DEFAULT_WORDS_PATH)

    if not os.path.exists(glove_path):
        raise SystemExit(
            f"DAT scoring needs GloVe vectors, which are not shipped with this repo.\n"
            f"Looked for: {glove_path}\n"
            f"Download {GLOVE_DOWNLOAD_URL} and unzip glove.840B.300d.txt into\n"
            f"  {DEFAULT_GLOVE_PATH.parent}\n"
            f"or point --glove-path / $CREATIVITYPRISM_GLOVE_PATH at an existing copy."
        )

    model = Model(glove_path, words_path)
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
    # The historical default rewrites the path in place, which silently overwrites the
    # inference file when it contains no "inference" segment. Callers should pass --output-path.
    dat_save_path = output_path or dat_inference_result_path.replace("inference", "evaluation")
    if os.path.abspath(dat_save_path) == os.path.abspath(dat_inference_result_path):
        raise SystemExit(
            f"Refusing to overwrite the inference file: {dat_inference_result_path}\n"
            "Pass --output-path explicitly."
        )
    os.makedirs(os.path.dirname(dat_save_path) or ".", exist_ok=True)

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