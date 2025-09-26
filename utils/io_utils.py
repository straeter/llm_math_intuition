import json

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    return data

def load_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(result, output_file):
    """Write result to file and update processed IDs set"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

