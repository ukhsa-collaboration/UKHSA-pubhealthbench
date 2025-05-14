import os
import openai
from anthropic import Anthropic
from openai import OpenAI
import anthropic
import json
import re
import random
from tqdm import tqdm
import time
from datasets import load_dataset
import argparse

from compute_accuracy import extract_answer, process_judgement_json
from prompt_lib.prompts import mcqa_zero_shot, freeform_zero_shot, freeform_judge_cot

API_KEY = os.environ.get("API_KEY", "")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "")

OPENAI_NON_REASONING = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini-2024-07-18"]
OPENAI_REASONING = ["o1-preview", "o3-mini"]
ANTHROPIC_MODELS = ["claude-3-7-sonnet-20250219"]
GOOGLE_MODELS = ["gemini-2.0-flash"]

random.seed(12345)


def get_client(model_name, api_key):
    if model_name in OPENAI_REASONING + OPENAI_NON_REASONING:
        openai.api_key = api_key
        client = OpenAI(api_key=openai.api_key)
    elif model_name in GOOGLE_MODELS:
        openai.api_key = api_key
        client = OpenAI(api_key=openai.api_key,
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    elif model_name in ANTHROPIC_MODELS:
        client = anthropic.Anthropic(
            api_key=api_key,
        )
    else:
        client = None
        print("For other model API calls, please implement the client definition method yourself.")
    return client


def call_api(client, system_prompt, prompt_body, model_name):
    start = time.time()
    if model_name in OPENAI_NON_REASONING + GOOGLE_MODELS:
        message_text = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_body}
        ]
        completion = client.chat.completions.create(
            model=model_name,
            messages=message_text,
            temperature=0,
            max_tokens=4000,
            top_p=1,
        )
        result = completion.choices[0].message.content
    elif model_name in OPENAI_REASONING:
        message_text = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_body}
        ]
        completion = client.chat.completions.create(
            model=model_name,
            messages=message_text,
            reasoning_effort='low'
        )
        result = completion.choices[0].message.content
    elif model_name in ANTHROPIC_MODELS:
        message_text = [
            {"role": "user", "content": prompt_body}
        ]
        anthropic_system_prompt = system_prompt if system_prompt and system_prompt.strip() else None
        message = client.messages.create(
            model=model_name,
            max_tokens=4000,
            system=anthropic_system_prompt,
            messages=message_text,
            temperature=0.0,
            top_p=1,
        )
        result = message.content[0].text
    else:
        print("For other model API calls, please implement the request method yourself.")
        result = None
    print("cost time", time.time() - start)
    return result


def load_pubhealthbench(subset):
    dataset = load_dataset("Joshua-Harris/PubHealthBench")
    test_df, val_df, reviewed_df = dataset["test"], dataset["validation"], dataset["reviewed"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    reviewed_df = preprocess(reviewed_df)
    if subset == "reviewed":
        return reviewed_df, val_df
    elif subset == "full":
        return test_df, val_df
    else:
        raise ValueError("Invalid subset")


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res


def single_request(client, single_question, prompt_template, exist_result, extract_func, model_name):
    exist = True
    q_id = single_question["question_id"]
    for each in exist_result:
        if q_id == each["question_id"] and single_question["question"] == each["question"]:
            pred = extract_func(each["model_outputs"])
            return pred, each["model_outputs"], exist
    exist = False
    system_prompt = prompt_template['system_prompt']
    prompt_body = prompt_template['prompt_body'].format(**single_question)
    try:
        response = call_api(client, system_prompt, prompt_body, model_name)
        response = response.replace('**', '')
    except Exception as e:
        print("error", e)
        return None, None, exist
    pred = extract_func(response)
    return pred, response, exist


def update_result(output_res_path):
    category_record = {}
    res = []
    success = False
    while not success:
        try:
            if os.path.exists(output_res_path):
                with open(output_res_path, "r") as fi:
                    res = json.load(fi)
                    for each in res:
                        category = each["category"]
                        if category not in category_record:
                            category_record[category] = {"corr": 0.0, "wrong": 0.0}
                        if each["pred"] == each["answer"]:
                            category_record[category]["corr"] += 1
                        else:
                            category_record[category]["wrong"] += 1
            success = True
        except Exception as e:
            print("Error", e, "sleep 2 seconds")
            time.sleep(2)
    return res, category_record


def merge_result(res, curr):
    merged = False
    for i, single in enumerate(res):
        if single["question_id"] == curr["question_id"] and single["question"] == curr["question"]:
            res[i] = curr
            merged = True
    if not merged:
        res.append(curr)
    return res


def evaluate(subjects, subset, setup):
    client = get_client(args.model_name, API_KEY)
    if setup == "freeform":
        judge_client = get_client(args.judge_model_name, JUDGE_API_KEY)
    test_df, dev_df = load_pubhealthbench(subset=subset)
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)
    for subject in subjects:
        test_data = test_df[subject]
        output_res_path = os.path.join(args.output_dir, subject + f"_{setup}" + "_result.json")
        output_summary_path = os.path.join(args.output_dir, subject + f"_{setup}" + "_summary.json")
        res, category_record = update_result(output_res_path)

        for each in tqdm(test_data):
            category = subject
            if setup == "mcqa":
                label = each["answer"]
                pred, response, exist = single_request(client, each, mcqa_zero_shot, res,
                                                       extract_func=extract_answer, model_name=args.model_name)
            elif setup == "freeform":
                # Correct label is judge says valid
                label = True
                each["answer"] = label
                print('LLM Free Form Response...')
                pred, response, exist = single_request(client, each, freeform_zero_shot, res,
                                                       extract_func=lambda x: x, model_name=args.model_name)
                # Set judge inputs
                each['ground_truth_answer'] = each['options'][each["answer_index"]]
                each['given_answer'] = response
                print('Judge Evaluation...')
                pred, response, exist = single_request(judge_client, each, freeform_judge_cot, res,
                                                       extract_func=process_judgement_json,
                                                       model_name=args.judge_model_name)
                print(pred, response)
            else:
                raise ValueError('Invalid setup')
            if response is not None:
                res, category_record = update_result(output_res_path)
                if category not in category_record:
                    category_record[category] = {"corr": 0.0, "wrong": 0.0}
                each["pred"] = pred
                each["model_outputs"] = response
                merge_result(res, each)
                if pred is not None:
                    if pred == label:
                        category_record[category]["corr"] += 1
                    else:
                        category_record[category]["wrong"] += 1
                else:
                    category_record[category]["wrong"] += 1
                save_res(res, output_res_path)
                save_summary(category_record, output_summary_path)
                res, category_record = update_result(output_res_path)
        save_res(res, output_res_path)
        save_summary(category_record, output_summary_path)


def save_res(res, output_res_path):
    temp = []
    exist_q_id = []
    for each in res:
        if each["question_id"] not in exist_q_id:
            exist_q_id.append(each["question_id"])
            temp.append(each)
        else:
            continue
    res = temp
    with open(output_res_path, "w") as fo:
        fo.write(json.dumps(res))


def save_summary(category_record, output_summary_path):
    total_corr = 0.0
    total_wrong = 0.0
    for k, v in category_record.items():
        if k == "total":
            continue
        cat_acc = v["corr"] / (v["corr"] + v["wrong"])
        category_record[k]["acc"] = cat_acc
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    acc = total_corr / (total_corr + total_wrong)
    category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
    with open(output_summary_path, "w") as fo:
        fo.write(json.dumps(category_record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="eval_results/")
    parser.add_argument("--model_name", "-m", type=str, default="gpt-4o",
                        choices=OPENAI_REASONING + OPENAI_NON_REASONING + ANTHROPIC_MODELS + GOOGLE_MODELS)
    parser.add_argument("--judge_model_name", "-j", type=str, default=None,
                        choices=OPENAI_REASONING + OPENAI_NON_REASONING + ANTHROPIC_MODELS + GOOGLE_MODELS)
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all")
    parser.add_argument("--subset", "-t", type=str, default="reviewed", choices=["full", "reviewed"])
    parser.add_argument("--setup", "-p", type=str, default="mcqa", choices=["mcqa", "freeform"])
    assigned_subjects = []
    args = parser.parse_args()
    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = args.assigned_subjects.split(",")
    os.makedirs(args.output_dir, exist_ok=True)
    print(assigned_subjects, args.subset, args.setup, args.judge_model_name, args.model_name)
    evaluate(assigned_subjects, args.subset, args.setup)
