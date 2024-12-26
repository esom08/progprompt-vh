import json

test_tasks = []
with open(f"progprompt-vh/data/new_env/env1_annotated.json", 'r') as f:
            for line in f.readlines():
                test_tasks.append(list(json.loads(line).keys())[0])
print(f"{test_tasks=}")
