#!/usr/bin/env python3
import os
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIR = REPO_ROOT / "torch_ttnn/cpp_extension/third-party/tt-metal/ttnn/cpp/ttnn/operations"
OUT_FILE = REPO_ROOT / "ttnn_ops_implemented.txt"


def iter_source_files(root: Path):
    exts = {".hpp", ".h", ".cpp", ".cc", ".cxx"}
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if Path(fname).suffix in exts:
                yield Path(dirpath) / fname


def find_registrations(text: str):
    # Capture registrations like: ttnn::register_operation<"ttnn::foo", ...>
    pattern = re.compile(r'ttnn::register_operation<\s*"([^"]+)"\s*,', re.MULTILINE | re.DOTALL)
    for m in pattern.finditer(text):
        yield m


def find_constexpr_name(lines, match_start_index):
    # Search up to 3 lines above the match start for: constexpr auto NAME =
    # Compute the line index of match_start_index first
    cum = 0
    line_index = 0
    for i, line in enumerate(lines):
        cum += len(line)
        if cum > match_start_index:
            line_index = i
            break
    start_search = max(0, line_index - 3)
    constexpr_rx = re.compile(r'constexpr\s+auto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=')
    for i in range(line_index, start_search - 1, -1):
        m = constexpr_rx.search(lines[i])
        if m:
            return m.group(1)
    return None


def main():
    results = []

    for path in sorted(iter_source_files(SEARCH_DIR)):
        rel = path.relative_to(REPO_ROOT)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lines = [ln + ("\n" if not ln.endswith("\n") else "") for ln in text.splitlines()]
        for m in find_registrations(text):
            op_name = m.group(1)
            # Compute line number (1-based)
            line_no = text.count("\n", 0, m.start()) + 1
            var_name = find_constexpr_name(lines, m.start())
            results.append((str(rel), line_no, op_name, var_name))

    # Sort by op name then file
    results.sort(key=lambda x: (x[2], x[0], x[1]))

    with OUT_FILE.open("w", encoding="utf-8") as f:
        f.write("TTNN operations registered ({} matches)\n\n".format(len(results)))
        for rel, line_no, op_name, var_name in results:
            f.write(f"{rel}:{line_no}\n")
            f.write(f"  op: \"{op_name}\"\n")
            if var_name:
                f.write(f"  var: {var_name}\n")
            f.write("\n")

    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()


