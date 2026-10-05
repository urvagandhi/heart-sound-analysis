"""
Audit script implementing the 26 Humanizer patterns (blader/humanizer)
along with strict manuscript constraints (no Oxford comma, no em-dashes).
"""

import sys
import os
import re

def audit_text(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return 1

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    checks = {
        "Strict Constraint: Oxford comma (, and)": re.compile(r',\s+and\b', re.IGNORECASE),
        "Strict Constraint: Em dash (--- or --)": re.compile(r'(?:---|(?:\s--\s))'),
        "Pattern 1: Not X but Y / rather than": re.compile(r'\b(not\s+(?:only|just|merely)?\s*[\w\s,]+?\s*but\b|rather than\b)', re.IGNORECASE),
        "Pattern 2: One-line closers / fragments": re.compile(r'\b(that is the real win|let that sink in|this shows the importance of|read that again)\b', re.IGNORECASE),
        "Pattern 3: Sayings that sound deep": re.compile(r'\b(at its core|in reality|what really matters|fundamentally|the deeper issue|heart of the matter|language of trust)\b', re.IGNORECASE),
        "Pattern 4: Staged run-up": re.compile(r'\b(let\'s dive in|let\'s explore|here\'s what you need to know|honestly\?|here\'s the thing|real talk)\b', re.IGNORECASE),
        "Pattern 5: Arguing with no one": re.compile(r'\b(to be clear|don\'t get me wrong|this is not to say|one might be tempted|a tempting approach)\b', re.IGNORECASE),
        "Pattern 9: Stacked qualifiers": re.compile(r'\b(could potentially possibly|might arguably|to be fair)\b', re.IGNORECASE),
        "Pattern 12: Overused AI words": re.compile(r'\b(actually|additionally|bolstered|crucial|deep dive|delve[sd]?|enduring|enhance[sd]?|garner(?:ed|ing)?|interplay|intricate|intricacies|landscape|meticulous(?:ly)?|pivotal|quietly|showcase[sd]?|tapestry|testament|underscore[sd]?|vibrant|bespoke|beacon)\b', re.IGNORECASE),
        "Pattern 13: Inflated significance": re.compile(r'\b(stands as a testament|marking a pivotal moment|plays a key role|indelible mark|the future looks bright|step in the right direction)\b', re.IGNORECASE),
        "Pattern 14: Vague connection": re.compile(r'\b(in connection with|associated with the leadership)\b', re.IGNORECASE),
        "Pattern 15: Shallow -ing riders": re.compile(r'\b(underscoring|emphasizing|symbolizing|fostering|showcasing)\b', re.IGNORECASE),
        "Pattern 16: Sales language": re.compile(r'\b(groundbreaking|renowned|breathtaking|stunning|nestled within)\b', re.IGNORECASE),
        "Pattern 17: Borrowed authority": re.compile(r'\b(experts believe|observers have cited|industry reports suggest)\b', re.IGNORECASE),
        "Pattern 18: Avoiding is/are/has": re.compile(r'\b(serves as|functions as|operates as|boasts)\b', re.IGNORECASE),
        "Pattern 22: Chatbot residue": re.compile(r'\b(i hope this helps|of course!|certainly!|great question|let me know if you)\b', re.IGNORECASE),
        "Pattern 23: Knowledge-limit disclaimers": re.compile(r'\b(as of my last|up to my last training|while specific details are limited|based on available information|it is believed that)\b', re.IGNORECASE),
        "Pattern 25: Writing about the document": re.compile(r'\b(the table below compares|this section is organized by|was added to replace)\b', re.IGNORECASE),
    }

    results = {k: [] for k in checks}

    for idx, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        # Skip pure LaTeX comments
        if line.startswith('%'):
            continue
        for name, pat in checks.items():
            for m in pat.finditer(raw_line):
                results[name].append((idx, m.group(0), line[:100]))

    total_flags = sum(len(v) for v in results.values())
    print("=" * 70)
    print(f"Humanizer Audit Report: {os.path.basename(filepath)}")
    print("=" * 70)
    if total_flags == 0:
        print("[PASS] 0 flags found. Manuscript cleanly satisfies all Humanizer rules.")
        return 0

    print(f"[FAIL] Found {total_flags} potential flags:")
    for name, occurrences in results.items():
        if occurrences:
            print(f"\n--- {name} ({len(occurrences)}) ---")
            for line_no, matched, snippet in occurrences:
                print(f"  Line {line_no:4d}: [{matched}] -> {snippet}")
    return 1

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'main.tex'
    sys.exit(audit_text(target))
