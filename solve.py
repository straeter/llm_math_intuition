import argparse
import asyncio
import os

from dotenv import load_dotenv
from pydantic import BaseModel

from utils.io_utils import load_jsonl, save_jsonl
from utils.llm import run_completion

load_dotenv()


def get_system_prompt(reasoning=False):
    reasoning_str = """Your output should look like this (after the "Answer:"):
<answer>|||<your reasoning why you probably came up with this answer>""" if reasoning else ""

    return """
You have to solve / estimate math problems WITHOUT reasoning / thinking / step by step method.
It is very important that you give the answer (without any reasoning, tools etc.) RIGHT AWAY.
Even if you do not know the answer, try to guess/estimate it IMMEDIATELY.
To make sure you are only giving the answer, we have limited the number of tokens to only 20.
Start your reply with 'The solution is'
""" + reasoning_str


class AnswerModel(BaseModel):
    answer: str


async def solve_single(
        query, model, max_tokens=30, reasoning=False
):
    question = query.get("question")

    msgs = [
        {"role": "user" if model.startswith("claude") else "system", "content": get_system_prompt(reasoning)},
        {"role": "user", "content": question + " Answer:"}
    ]

    params = dict(
        model=model,
        # response_model=AnswerModel,
        messages=msgs,
        json_mode=False,
        max_tokens=max_tokens,
    )

    if model.startswith("gpt-5"):
        params["reasoning"] = {"effort": "minimal"}
        params["text"] = {"verbosity": "low"}

    else:
        params["temperature"] = 0.0

    response = await run_completion(**params)

    split = response.split("|||")
    res = split[0].strip()
    res = res.replace("\n", " ").replace("Answer:", "").replace("The solution is:", "").replace("The solution is",
                                                                                                "").strip()
    query['response'] = res
    query['reasoning'] = split[1].strip() if len(split) > 1 else ""
    query['model'] = model

    response_file = f"data/responses_{model}.jsonl"
    save_jsonl(query, response_file)

    return res


async def solve_sem(query, model, sem):
    async with sem:  # limit concurrency
        return await solve_single(query=query, model=model)


async def solve_all(model: str, limit: int = None):
    questions = load_jsonl("data/questions.jsonl")
    if limit:
        questions = questions[:limit]

    if os.path.exists(f"data/responses_{model}.jsonl"):
        questions_solved = load_jsonl(f"data/responses_{model}.jsonl")
        questions_solved_ids = {q['id'] for q in questions_solved}
        questions = [q for q in questions if q['id'] not in questions_solved_ids]

    sem = asyncio.Semaphore(5)  # max 10 concurrent jobs
    tasks = []
    for q in questions:
        task = asyncio.create_task(solve_sem(q, model, sem))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gpt-5-mini", help='Model to evaluate')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of questions to evaluate (especially for testing)')
    args = parser.parse_args()

    asyncio.run(solve_all(model=args.model, limit=args.limit))
