
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