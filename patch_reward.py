from pathlib import Path
import re
import sys

# Find game file
path = Path("snake/game.py") if Path("snake/game.py").exists() else (Path("game.py") if Path("game.py").exists() else None)
if path is None:
    print("ERROR: Could not find snake/game.py or game.py. Run from repo root.")
    sys.exit(1)

text = path.read_text(encoding="utf-8")

# Ensure math import exists (in case reward uses sqrt somewhere)
if not re.search(r"(?m)^import\s+math\s*$", text):
    imports = list(re.finditer(r"(?m)^(from\s+\S+\s+import\s+.*|import\s+.*)\s*$", text))
    if imports:
        insert_at = imports[-1].end()
        text = text[:insert_at] + "\nimport math\n" + text[insert_at:]
    else:
        text = "import math\n" + text

lines = text.splitlines(True)
changed = 0

# 1) Fix "repeat" logic: after incrementing consecutive_moves[key], reset other keys to 0
inc_patterns = [
    re.compile(r"^(\s*)self\.consecutive_moves\[(.+?)\]\s*\+=\s*1\s*$"),
    re.compile(r"^(\s*)self\.consecutive_moves\[(.+?)\]\s*=\s*self\.consecutive_moves\.get\(\s*(.+?)\s*,\s*0\s*\)\s*\+\s*1\s*$"),
]

i = 0
while i < len(lines):
    line = lines[i]
    m = None
    key_expr = None
    indent = None

    for pat in inc_patterns:
        mm = pat.match(line)
        if mm:
            m = mm
            indent = mm.group(1)
            # key expression is group(2) for both patterns
            key_expr = mm.group(2).strip()
            break

    if m and indent and key_expr:
        # Look ahead a few lines to see if reset loop already exists
        window = "".join(lines[i+1:i+10])
        already_has_reset = ("for " in window and "consecutive_moves" in window and "keys()" in window and "self.consecutive_moves" in window)

        if not already_has_reset:
            insert = [
                f"{indent}for k in list(self.consecutive_moves.keys()):\n",
                f"{indent}    if k != {key_expr}:\n",
                f"{indent}        self.consecutive_moves[k] = 0\n",
            ]
            lines[i+1:i+1] = insert
            changed += 1
            i += len(insert)

    i += 1

new_text = "".join(lines)

# 2) If repeat penalty was being multiplied by the count, normalize it to a flat penalty
new_text2 = re.sub(
    r"reward\s*\+=\s*config\.PENALTY_REPEAT\s*\*\s*self\.consecutive_moves\[[^\]]+\]",
    "reward += config.PENALTY_REPEAT",
    new_text
)
if new_text2 != new_text:
    changed += 1
new_text = new_text2

path.write_text(new_text, encoding="utf-8")
print(f"Patched: {path} (changes applied: {changed})")
