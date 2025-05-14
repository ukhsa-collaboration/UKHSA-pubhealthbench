import glob
import sys
import json
import re
import random



def extract_answer(text, level='l2'):
    if level == 'l1':
        pattern = r"answer is \(?([A-G])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return None
    elif level == 'l2':
        pattern = r"answer is \(?([A-G])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-G])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)
    

def extract_final(text):
    pattern = r"\b[A-G]\b(?!.*\b[A-G]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

def process_judgement_json(llm_judge_response, *args, **kwargs):
    cleaned_json_str = clean_response_string(llm_judge_response)
    try:
        parsed = json.loads(cleaned_json_str)
        return parsed.get('predicted_correct', None)
    except (json.JSONDecodeError, TypeError):
        return None

def clean_response_string(response):
    cleaned_string = response.replace("```", "").replace("json", "").replace("\n", "").replace("True", "true").replace(
        "False", "false")
    return cleaned_string


if __name__ == '__main__':
    assert len(sys.argv) > 1, 'You need to pass the directory'
    path = sys.argv[1]
    random.seed(12345)

    succ_total_l1, fail_total_l1 = 0, 0
    succ_total_l2, fail_total_l2 = 0, 0
    for name in glob.glob(path + '/*_result.json'):
        if 'freeform' in name:
            extract_func = process_judgement_json
        elif 'mcqa' in name:
            extract_func = extract_answer
        else:
            raise Exception('Unknown setup')
        print('Level 1 regex' + '==' * 20)
        succ, fail = 0, 0
        with open(name, 'r') as f:
            entries = json.load(f)
            for e in entries:
                pred = extract_func(e['model_outputs'], 'l1')
                if pred == e['answer']:
                    succ += 1
                    succ_total_l1 += 1
                else:
                    fail += 1
                    fail_total_l1 += 1
        print(name, succ / (succ + fail))

        print('Level 2 regex' + '==' * 20)
        succ, fail = 0, 0
        with open(name, 'r') as f:
            entries = json.load(f)
            for e in entries:
                pred = extract_func(e['model_outputs'], 'l2')
                if pred == e['answer']:
                    succ += 1
                    succ_total_l2 += 1
                else:
                    fail += 1
                    fail_total_l2 += 1
        print(name, succ / (succ + fail))

        print()
    print((succ_total_l1 + fail_total_l1))
    print('Overall L1: ', succ_total_l1 / (succ_total_l1 + fail_total_l1))
    print('Overall L2: ', succ_total_l2 / (succ_total_l2 + fail_total_l2))
