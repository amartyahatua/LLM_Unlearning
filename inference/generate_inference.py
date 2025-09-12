import pandas as pd
from tqdm import tqdm

def generate_inference(model, tokenizer, question_string):

    alpaca_prompt = """Answer the following question:
    ### Question:
    {}

    ### Answer:
    {}"""

    inputs = tokenizer(
      [
          alpaca_prompt.format(
              f"{question_string}",
              #f"Has Hsiao Yun-Hwa's mother's unemployment played a role in her writings?",
              "" # The model will fill in the answer here
          )
      ], return_tensors="pt").to("cuda")
    outputs = model.generate(
      input_ids = inputs.input_ids,
      attention_mask = inputs.attention_mask,
      max_new_tokens = 64, use_cache = True
    )

  # Decode the entire output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Define the marker for the answer
    answer_marker = "### Answer:\n"
    question_marker = "### Question:"
    # Find the index of the answer marker
    answer_start_index = full_output.find(answer_marker)

    # Check if the marker was found
    if answer_start_index != -1:
      # Extract the text after the marker
      answer_only = full_output[answer_start_index + len(answer_marker):].strip()
      #print(answer_only)
    else:
      print("Could not find the answer marker in the output for the first generation.")
      print(full_output) # Print the full output for debugging

    return answer_only.split("\n\n")[0]

def get_results(model_name, dataset_inference, model, tokenizer):
    df_results = pd.DataFrame()
    dataset_inference = dataset_inference()
    for i in tqdm(range(len(dataset_inference))):
        # if i > 1000:
        #   break
        question_string = dataset_inference['question'][i]
        actual = dataset_inference['answer'][i]
        # question_string = "Where is Boston?"
        answer = generate_inference(model, tokenizer, question_string)
        # print(f"Question: {question_string}\nAnswer: {answer} \n Actual answer: {actual}")
        df_results.loc[i, 'question'] = question_string
        df_results.loc[i, 'answer'] = answer
        df_results.loc[i, 'actual'] = actual
    df_results.to_csv(f'result/{model_name}_result.csv')
