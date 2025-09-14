import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from datetime import datetime
from llms import get_llm_provider
from pydantic import BaseModel
from typing import Optional

SYSTEM_PROMPT = """You are an expert in venture capital tasked with identifying successful founders from their unsuccessful counterparts. 
All founders under consideration are sourced from LinkedIn and Crunchbase profiles of companies that have raised between $100K and $4M in funding. 
A successful founder is defined as one whose company has achieved either a total funding of over $500M or an exit/IPO valued at over $500M."""

USER_PROMPT = """Given the following founder description:
       {founder_description},
       please output a json string with two keys; 
       1. prediction: 'Yes' or 'No' corresponding to whether or not the founder will be successful.
       2. reasoning: a short explanation for your prediction (at most 100 words).
    DO NOT return anything else"""

class Prediction(BaseModel):
    prediction: Optional[str] = None
    reasoning: Optional[str] = None


def vanilla_llm_testing(input_file: str, provider: str, model: str):
    founders_data = pd.read_csv(input_file)
    args_list = [(row, provider, model) for _, row in founders_data.iterrows()]

    num_processes = cpu_count()
    if provider == "openai":
        with Pool(processes=num_processes) as pool:
            predictions = pool.map(_get_prediction_openai, args_list)
    else:
        raise ValueError(f"Provider {provider} not supported")

    output_df = pd.DataFrame(columns=['founder_uuid', 'name', 'success', 'prediction', 'anonymised_prose'])
    output_df['founder_uuid'] = founders_data['founder_uuid']
    if 'name' in founders_data.columns:
        output_df['name'] = founders_data['name']
    output_df['success'] = founders_data['success']
    output_df['prediction'] = predictions
    output_df['anonymised_prose'] = founders_data['anonymised_prose']

    os.makedirs("vanilla_llm_testing_results", exist_ok=True)
    output_path = os.path.join(
        "vanilla_llm_testing_results",
        f'{model}_{input_file.split(".")[0].split("/")[-1]}_{datetime.now().strftime("%m-%d_%H-%M-%S")}.csv'
    )
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")


def _get_prediction_openai(args):
    row, provider, model = args
    llm_provider = get_llm_provider(provider, model=model)
    founder_description = row["anonymised_prose"]
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT.format(founder_description=founder_description)
    prediction = llm_provider.get_llm_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=1.0,
    )
    if 'name' in row:
        print(f"{row['name']}: {prediction}")
    else:
        print(f"Prediction: {prediction}")
    return prediction


def main():
    file = "vcbench_final_public_sample100.csv"
    vanilla_llm_testing(file, "openai", "gpt-4o-mini")
    
        
if __name__ == "__main__":
    main()