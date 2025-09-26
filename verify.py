import argparse
import asyncio
import json
import os
from typing import Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from utils.io_utils import load_jsonl, save_jsonl
from utils.llm import run_completion

load_dotenv()


def get_system_prompt():
    return """
You should verify if the provided attempts (field "response") to the math problems are correct (correct solution is in field "answer"), as they might be paraphrased etc.
Sometimes you have to compare a fraction with a decimal number, or a rounded number with a precise one. In that case, consider the answer correct if it is reasonably close (like they coincide on the first 2 non-zero values).
Make sure to go throguh all the ids.

Return your answer in the following JSON format:
{
"verifications": {
"<id1>": "true" | "false",
"<id2>": "true" | "false",
...
}
}
"""


#
class AnswerModel(BaseModel):
    verifications: Dict[str, str] = Field(default_factory=dict)


async def verify_batch(
        batch, model, max_tokens=10000, reasoning=False):
    batch_json = json.dumps(batch)

    msgs = [
        {"role": "system", "content": get_system_prompt(reasoning)},
        {"role": "user", "content": batch_json}
    ]

    params = dict(
        model="gpt-5-mini",  # verification model remains the same
        # response_model=AnswerModel,
        messages=msgs,
        json_mode=True,
        max_tokens=None
        # temperature=0,
    )

    response_file = f"data/responses_{model}_verified.jsonl"

    response_json = await run_completion(**params)

    verifications = response_json["verifications"]
    for query in batch:
        id = query.get("id")
        verification = verifications.get(str(id))
        query['correct'] = True if verification in ["True", "true", True] else False
        save_jsonl(query, response_file)

    return response_json


async def verify_batch_sem(batch, model, sem):
    async with sem:  # limit concurrency
        return await verify_batch(batch=batch, model=model)


async def verify_all(model: str, limit: int = None):
    questions = load_jsonl(f"data/responses_{model}.jsonl")
    if limit:
        questions = questions[:limit]

    if os.path.exists(f"data/responses_{model}_verified.jsonl"):
        questions_verified = load_jsonl(f"data/responses_{model}_verified.jsonl")
        verified_ids = set([q['id'] for q in questions_verified])
        questions = [q for q in questions if q['id'] not in verified_ids]

    sem = asyncio.Semaphore(5)  # max 10 concurrent jobs
    tasks = []
    batch_size = 20
    # loop through batches of questions:
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        task = asyncio.create_task(verify_batch_sem(batch, model, sem))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gemini-2.5-pro", help='Model to evaluate')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of responses to verify (especially for testing)')
    args = parser.parse_args()
    asyncio.run(verify_all(model=args.model, limit=args.limit))
