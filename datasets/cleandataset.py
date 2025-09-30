import json
import re

INPUT_FILE = "datasets/unified_dataset.json"
OUTPUT_FILE = "datasets/cleaned_unified_dataset.json"

required_fields = ["dataset", "task", "input", "target"]
seen_pairs = set()

def normalize_line(line: str) -> str:
    """Replace unusual line terminators with standard \n and strip spaces."""
    for char in ['\x0b', '\x0c', '\u2028', '\u2029']:
        line = line.replace(char, '\n')
    return line.strip()

def clean_task(task_str: str) -> str:
    """Clean task string by removing unwanted symbols and normalizing."""
    task_str = re.sub(r'[^a-zA-Z0-9_]', '_', task_str)  # keep letters, digits, underscore
    task_str = re.sub(r'_+', '_', task_str)  # collapse multiple underscores
    return task_str.strip('_')

with open(INPUT_FILE, "r", encoding="utf-8") as f_in, open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    f_out.write("[\n")
    first_entry = True

    for line in f_in:
        line = normalize_line(line)
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip malformed lines

        # Check required fields
        if not all(field in entry and entry[field] for field in required_fields):
            continue

        # Clean fields
        entry['input'] = entry['input'].strip()
        entry['target'] = entry['target'].strip()
        entry['task'] = clean_task(entry['task'])

        # Remove duplicates
        identifier = (entry['input'], entry['target'])
        if identifier in seen_pairs:
            continue
        seen_pairs.add(identifier)

        # Write to output
        if not first_entry:
            f_out.write(",\n")
        else:
            first_entry = False
        json.dump(entry, f_out, ensure_ascii=False)

    f_out.write("\n]")

print(f"Cleaned dataset saved to '{OUTPUT_FILE}'")
print(f"Total records after cleaning: {len(seen_pairs)}")
