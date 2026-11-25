"""Install the trainer bridge into a disposable, pinned upstream checkout."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

PIN = "fd83f5b6e609b61b0b10777e32c74343e1a81b41"
HERE = Path(__file__).resolve().parent


def prepare(root, windows):
    root = Path(root).resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != PIN:
        raise ValueError(f"Expected upstream {PIN}, found {head}")
    if windows:
        patch = str(HERE / "windows-coff.patch")
        if subprocess.run(["git", "apply", "--reverse", "--check", patch], cwd=root,
                          capture_output=True).returncode:
            subprocess.run(["git", "apply", patch], cwd=root, check=True)
    groups = json.loads((root / "data/maps/map_groups.json").read_text())
    counts = ", ".join(str(len(groups[g])) for g in groups["group_order"])
    (root / "include/rl_map_counts.h").write_text(f"static const int rlMapCounts[] = {{{counts}}};\n")
    # The asset generator's recursive make already has tools and generated maps.
    # Propagating SETUP_PREREQS=1 makes upstream's "make generated" call itself.
    assets = root / "tools/generate_wasm_assets.py"
    original = assets.read_text(encoding="utf-8")
    assets.write_text(original.replace("'SETUP_PREREQS=1', target", "'SETUP_PREREQS=0', target"),
                      encoding="utf-8", newline="\n")
    menu = (root / "src/main_menu.c").read_text(encoding="utf-8")
    tasks = sorted(set(re.findall(r"^static void (Task_NewGameBirchSpeech\w*)\(u8\);", menu, re.M)))
    if not tasks:
        raise ValueError("Could not identify Birch introduction tasks")
    wrappers = {
        "title_screen.c": "int RlTitlePhase(void) { return gMain.callback2 == MainCB2 || gMain.callback2 == CB2_InitTitleScreen; }",
        "naming_screen.c": "int RlNamingPhase(void) { return gMain.callback2 == CB2_NamingScreen ? 4 : 0; }",
        "main_menu.c": """int RlMenuPhase(void) {
    if (gMain.callback2 != CB2_MainMenu) return 0;
    for (int i = 0; i < NUM_TASKS; ++i) {
        if (gTasks[i].isActive && (%s)) return 3;
    }
    return 2;
}""" % " || ".join("gTasks[i].func == " + task for task in tasks),
    }
    for name, wrapper in wrappers.items():
        source = root / "src" / name
        marker = "\n/* EmeraldRL read-only callback identity */"
        base = source.read_text(encoding="utf-8").partition(marker)[0]
        source.write_text(base + marker + "\n#if NATIVE\n" + wrapper + "\n#endif\n", encoding="utf-8", newline="\n")
    shutil.copyfile(HERE / "observe.c", root / "src/rl_observe.c")
    shutil.copyfile(HERE / "bridge.c", root / "tools/native/rl_bridge.c")
    shutil.copyfile(HERE / "build_target.mk", root / "rl-build.mk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout")
    parser.add_argument("--windows", action="store_true")
    args = parser.parse_args()
    prepare(args.checkout, args.windows)
