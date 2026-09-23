from collections import defaultdict
from pathlib import Path
import random
import re


def discover_writers(root, writers=None):
    roots = [Path(item.strip()) for item in str(root).split(",") if item.strip()]
    selected = {str(w) for w in writers} if writers else None
    result = defaultdict(lambda: {"genuine": [], "forged": []})
    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    # Support the documented data/<writer>/{genuine,forged} layout and
    # datasets such as BHSig, which store G/F files directly in writer folders.
    for root in roots:
        candidates = [p for p in root.rglob("*") if p.is_dir()]
        for writer in candidates:
            files = [p for p in writer.iterdir() if p.is_file() and p.suffix.lower() in image_extensions]
            if not files:
                continue
            # Prefix writer IDs so writers 1, 2, ... from separate datasets
            # remain distinct identities when datasets are combined.
            writer_name = f"{root.name}_{writer.name}"
            if selected is not None and writer.name not in selected and writer_name not in selected:
                continue
            for path in files:
                parent = path.parent.name.lower()
                stem = path.stem.lower()
                if (parent in {"genuine", "original", "originals"}
                        or re.search(r"(?:^|[-_])g(?:[-_]|$)", stem)
                        or stem.startswith("original")):
                    result[writer_name]["genuine"].append(path)
                elif (parent in {"forged", "forgery", "forgeries"}
                      or re.search(r"(?:^|[-_])f(?:[-_]|$)", stem)
                      or stem.startswith("forger")):
                    result[writer_name]["forged"].append(path)
    return result


def create_pairs(root, writers=None, pairs_per_class=2000, seed=42):
    rng = random.Random(seed)
    data = discover_writers(root, writers)
    positive, negative = set(), set()
    genuine = [p for d in data.values() for p in d["genuine"]]
    for d in data.values():
        for a in d["genuine"]:
            for b in d["genuine"]:
                if a != b:
                    positive.add((str(a), str(b), 1))
            for b in d["forged"]:
                negative.add((str(a), str(b), 0))
    def select(items):
        items = list(items)
        if not items:
            return []
        return rng.sample(items, min(len(items), pairs_per_class)) if len(items) >= pairs_per_class else [items[i % len(items)] for i in range(pairs_per_class)]
    result = select(positive) + select(negative)
    rng.shuffle(result)
    print(f"Generated {len(result)} pairs ({sum(x[2] == 1 for x in result)} genuine, {sum(x[2] == 0 for x in result)} forged).")
    return result
