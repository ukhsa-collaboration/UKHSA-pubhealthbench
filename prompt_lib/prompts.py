mcqa_zero_shot = {
    'system_prompt': """You are an expert working for a Public Health agency.""",
    'prompt_body': """The following are multiple choice questions (with answers) about UK Government public health guidance.\n\nQuestion: This question relates to UK Health Security Agency (UKHSA) guidance that could be found on the gov.uk website as of 08/01/2025.\n\n{question}\nOptions:\n{options_formatted}\n\nProvide the letter (A, B, C, D, E, F, or G) of the correct answer. You should state "The answer is (X)", where the X contained in the brackets is the correct letter choice, make sure you include the brackets () around your final answer in your response. DO NOT provide any other information or text in your response.\n\nAnswer: """,
}

freeform_zero_shot = {
    'system_prompt': """You are an expert working for a Public Health agency.""",
    'prompt_body': """The following is a question about UK Government public health guidance.\n\nQuestion: This question relates to UK Health Security Agency (UKHSA) guidance.\n\n{question}\n\nPlease answer the question to the best of your knowledge.\n\nAnswer: """,
}


freeform_judge_cot = {
'system_prompt': """You are an expert in UK Public Health. You are going to evaluate whether a given answer to a public health guidance question is correct""",
'prompt_body': """You are tasked with evaluating whether a given answer is correct based on the ground truth answer and provided context. Carefully analyse the ground truth answer and context and determine whether the given answer correctly answers the question and aligns with the information given.\n---\n\n===========\nQuestion:\n===========\n{question}\n---\n\n===========\nContext:\n===========\n{retrieved_context_for_judge}\n---\n\n===========\nGround Truth Answer:\n===========\n{ground_truth_answer}\n---\n\n===========\nGiven Answer:\n===========\n{given_answer}\n---\n\nFor the given answer to be correct it must align with the ground truth without omitting any key details **and** any additional detail in the given answer must be seen in the provided context. Determine, with reasoning, whether the given answer is correct based on the ground truth answer and context. Give your response in the following json format: json```{{"reasoning": Why is the answer correct/incorrect, "predicted_correct": true or false}}`"""
}
