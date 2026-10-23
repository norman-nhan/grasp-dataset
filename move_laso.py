import json
import shutil
from pathlib import Path
import os

DATA_ROOT = Path(os.path.join('data', 'laso_grasp_only_affordance'))
CATALOG = Path("laso_object_types.json")


def main():
    categories = json.loads(CATALOG.read_text())

    for category, object_ids in categories.items():
        category_dir = DATA_ROOT / category
        category_dir.mkdir(parents=True, exist_ok=True)

        for object_id in object_ids:
            source = DATA_ROOT / object_id
            destination = category_dir / object_id

            if not source.is_dir():
                print(f"Missing: {source}")
                continue

            if destination.exists():
                print(f"Already exists: {destination}")
                continue

            shutil.move(str(source), str(destination))
            print(f"Moved: {source} -> {destination}")


if __name__ == "__main__":
    main()