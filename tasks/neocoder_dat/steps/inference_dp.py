"""Perform parallel-thread inference on generated DP dataset.
"""
import click
import os
from torch.utils.data import DataLoader

from src.generator.generator import CodeGenerator
from src.utils.configs import get_dp_inference_params

@click.command()
@click.option("--dataset-path", type=click.Path(exists=True), required=True, help="The path of the DP dataset.")
@click.option("--model-name", type=str, required=True, help="The name of the model.")
@click.option("--dp-rounds", type=int, default=3, help="The number of DP rounds.")
@click.option("--batch-size", type=int, default=2, help="Note the real batch size feeding to hf model is batch_size * dp_rounds.")
@click.option("--output-dir", type=click.Path(exists=False), required=True, help="The directory to save the generated codes.")
@click.option("--overwrite", is_flag=True, help="Overwrite the existing file.")
@click.option("--temperature", type=float, default=0.7, help="The temperature for sampling.")
@click.option("--top-p", type=float, default=1.0, help="The top p for sampling.")
@click.option("--max-tokens", type=int, default=512, help="The maximum number of tokens to generate.")

def main(
    dataset_path,
    model_name,
    dp_rounds,
    batch_size,
    output_dir,
    overwrite,
    temperature,
    top_p,
    max_tokens
):

    params = get_dp_inference_params(dataset_path,
                                     model_name,
                                     dp_rounds,
                                     batch_size,
                                     temperature,
                                     top_p,
                                     max_tokens
                                    )
    
    generator: CodeGenerator = params['generator']
    dataloader: DataLoader = params['dataloader']

    # difficulty level
    num_sample = len(dataloader.dataset)
    model_name = model_name.split('/')[1] if '/' in model_name else model_name

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{model_name}_sample={num_sample}_dp={dp_rounds}_{int(float(temperature)*100)}.json')

    generator.inference(dataloader,
                        save_path,
                        overwrite=overwrite)

if __name__ == "__main__":
    main()

