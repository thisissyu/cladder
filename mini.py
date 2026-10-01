import json
import re
import urllib.request
from pathlib import Path


QUESTIONS_PATH = Path("runs/mini/questions.json")
MODELS_PATH = Path("runs/mini/models.json")
RESULT_PATH = Path("runs/mini/result.json")

API_URL = "http://127.0.0.1:38000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3.5-0.8B"


def find_model(obj, model_id):
    """Find the model whose id matches model_id."""

    target = str(model_id)

    if isinstance(obj, dict):
        # models.json might itself be keyed by model_id
        if model_id in obj:
            return obj[model_id]
        if target in obj:
            return obj[target]

        # or model_id might be a field
        for key in ("model_id", "id"):
            if key in obj and str(obj[key]) == target:
                return obj

        for value in obj.values():
            if isinstance(value, (dict, list)):
                result = find_model(value, model_id)
                if result is not None:
                    return result

    elif isinstance(obj, list):
        for item in obj:
            result = find_model(item, model_id)
            if result is not None:
                return result

    return None


def find_only_background(obj):
    """Fallback for our mini dataset, which contains only one model."""

    found = []

    def walk(x):
        if isinstance(x, dict):
            if "background" in x:
                found.append(x)
            for value in x.values():
                walk(value)
        elif isinstance(x, list):
            for value in x:
                walk(value)

    walk(obj)

    if len(found) == 1:
        return found[0]

    return None


def call_vllm(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0,
        "max_tokens": 64,
    }

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Do not send localhost traffic through HTTP_PROXY/HTTPS_PROXY.
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({})
    )

    with opener.open(req) as response:
        result = json.load(response)

    return result["choices"][0]["message"]["content"]


def parse_answer(text):
    match = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
    if match is None:
        return None

    return match.group(1).lower()


def main():
    questions = json.loads(QUESTIONS_PATH.read_text())
    models = json.loads(MODELS_PATH.read_text())

    # We generated exactly one question.
    q = questions[0]

    meta = q.get("meta", {})
    model_id = meta.get("model_id")

    model = None

    if model_id is not None:
        model = find_model(models, model_id)

    # Mini dataset contains one model, so this is a convenient fallback.
    if model is None:
        model = find_only_background(models)

    if model is None:
        raise RuntimeError("Could not find the causal model metadata")

    background = model["background"]

    prompt = "\n\n".join([
        background,
        q["given_info"],
        q["question"],
        (
            "Reason step by step about the causal effect. "
            "Your final answer must start with Yes or No."
        ),
    ])

    print("=== Prompt ===")
    print(prompt)
    print()

    raw_output = call_vllm(prompt)
    predicted = parse_answer(raw_output)
    expected = q["answer"].lower()

    correct = predicted == expected

    print("=== Result ===")
    print(f"Raw model output : {raw_output!r}")
    print(f"Predicted        : {predicted}")
    print(f"Ground truth     : {expected}")
    print(f"Correct          : {correct}")

    if "groundtruth" in meta:
        print(f"Causal quantity  : {meta['groundtruth']}")

    result = {
        "question_id": q.get("question_id"),
        "prompt": prompt,
        "raw_output": raw_output,
        "predicted": predicted,
        "answer": expected,
        "correct": correct,
        "meta": meta,
    }

    RESULT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )

    print()
    print(f"Saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
