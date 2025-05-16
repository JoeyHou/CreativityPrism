import click
import os
from torch.utils.data import DataLoader

from src.generator.generator import CodeGenerator
from src.utils.configs import get_inference_params, DAT_PROMPT

@click.command()
@click.option("--model-name", type=str, required=True, help="The name of the model.")
@click.option("--repeat", type=int, default=100, help="The number of DP rounds.")
@click.option("--output-dir", type=click.Path(exists=False), required=True, help="The directory to save the generated codes.")
@click.option("--overwrite", is_flag=True, help="Overwrite the existing file.")
@click.option("--temperature", type=float, default=0.7, help="The temperature for sampling.")
@click.option("--top-p", type=float, default=1.0, help="The top p for sampling.")
@click.option("--max-tokens", type=int, default=512, help="The maximum number of tokens to generate.")

def main(
    model_name,
    repeat,
    output_dir,
    overwrite,
    temperature,
    top_p,
    max_tokens
):

    params = get_inference_params(model_name,
                                  temperature,
                                  top_p,
                                  max_tokens)
    
    generator: CodeGenerator = params['generator']

    model_name = model_name.split('/')[1] if '/' in model_name else model_name

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{model_name}_{int(float(temperature)*100)}.json')

    generator.inference(problem_statement=DAT_PROMPT,
                        save_path=save_path,
                        overwrite=overwrite,
                        repeat=repeat)

if __name__ == "__main__":
    main()

