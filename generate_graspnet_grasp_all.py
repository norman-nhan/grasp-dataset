from pathlib import Path
from generate_graspnet_grasp import main as generate

DATA_ROOT = Path("data")
OUTPUT_ROOT = Path("outputs/graspnet")
FAILED_LOG = OUTPUT_ROOT / "failed_objects.txt"


def catalog(data_root: Path) -> list[Path]:
    return sorted(data_root.glob("**/affordance.npz"))


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for npz_path in catalog(DATA_ROOT):
        object_id = npz_path.parent.relative_to(DATA_ROOT)
        output = OUTPUT_ROOT / object_id / "graspnet_grasps.json"

        # Skip already generated objects
        if output.is_file():
            print(f"Skipping (already generated): {object_id}")
            continue

        print(f"Generating grasps for {object_id}")

        try:
            generate([
                "--npz", str(npz_path),
                "--max-candidates", "200",
                "--output-json", str(output),
                "--output", str(output.with_suffix(".png")),
            ])

        except ValueError as e:
            print(f"Failed: {object_id} - {e}")

            # Save failed object and error
            with FAILED_LOG.open("a") as f:
                f.write(f"{object_id}: {e}\n")

            continue


if __name__ == "__main__":
    main()