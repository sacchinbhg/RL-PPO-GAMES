"""Copy the published source into the two-directory layout used by the experiments."""
import argparse
from pathlib import Path
import shutil


def materialize(destination):
    archive = Path(__file__).resolve().parent
    project = archive.parent
    runtime = destination / "outputs" / "EmeraldRL"
    sources = {}
    for file in project.iterdir():
        if file.is_file() and file.name != "local.json" and file.suffix in {".py", ".ps1", ".json", ".txt", ".md"}:
            sources[runtime / file.name] = file
    for name in ("native", "results", "starter-diagnostics/scripts"):
        for file in (project / name).rglob("*"):
            if file.is_file() and "__pycache__" not in file.parts:
                sources[runtime / file.relative_to(project)] = file
    for file in (archive / "work").rglob("*"):
        if file.is_file() and "__pycache__" not in file.parts:
            sources[destination / "work" / file.relative_to(archive / "work")] = file
    conflicts = [path for path, file in sources.items() if path.exists() and
                 (not path.is_file() or path.read_bytes() != file.read_bytes())]
    if conflicts:
        raise FileExistsError("Existing files differ:\n" + "\n".join(map(str, conflicts)))
    for path, file in sources.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, path)
    print(f"Staged {len(sources)} source and evidence files under {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    materialize(args.destination.resolve())
