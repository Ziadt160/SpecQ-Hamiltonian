import os
import glob
import re

# Files to scan
files = glob.glob("experiments/*.py") + glob.glob("src/**/*.py", recursive=True)

for file in files:
    # Skip seeds.py itself
    if "seeds.py" in file.replace("\\", "/"):
        continue
        
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Check if file has seed logic
    # Look for 'seed' or 'random_state' (case insensitive)
    if not re.search(r'\b(seed|random_state)\b', content, re.IGNORECASE):
        continue
        
    # Skip if already imported
    if "from src.utils.seeds import set_seed" in content:
        continue
        
    # Find last import statement to insert after it
    lines = content.split('\n')
    out_lines = []
    
    last_import_index = -1
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            last_import_index = i
            
    if last_import_index != -1:
        for i, line in enumerate(lines):
            out_lines.append(line)
            if i == last_import_index:
                out_lines.append("")
                out_lines.append("from src.utils.seeds import set_seed")
                # Following user's pattern in experiment_mnist.py
                out_lines.append("set_seed()")
    else:
        out_lines.append("from src.utils.seeds import set_seed")
        out_lines.append("set_seed()")
        out_lines.append("")
        out_lines.extend(lines)
        
    with open(file, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    
    print(f"Modified {file}")
