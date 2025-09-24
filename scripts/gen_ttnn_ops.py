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
    constexpr_rx = re.compile(r"constexpr\s+auto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=")
    for i in range(line_index, start_search - 1, -1):
        m = constexpr_rx.search(lines[i])
        if m:
            return m.group(1)
    return None


def find_unary_macro_registrations(text: str):
    # Capture macro usages like: REGISTER_UNARY_OPERATION(acos, ACOS)
    # and variations: REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(name, TYPE)
    # We only need the first argument (the variable/op name)
    pattern = re.compile(
        r"REGISTER_UNARY_OPERATION(?:_[A-Z_]+)?\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,",
        re.MULTILINE | re.DOTALL,
    )
    for m in pattern.finditer(text):
        yield m


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

        # Also detect macro-based unary registrations
        for m in find_unary_macro_registrations(text):
            macro_var = m.group(1)
            op_name = f"ttnn::{macro_var}"
            line_no = text.count("\n", 0, m.start()) + 1
            results.append((str(rel), line_no, op_name, macro_var))

    # Categorize results roughly following groupings used in the registration file
    def categorize(path_str: str, op_name: str):
        # Use sub-path under .../operations/ to drive grouping
        sub = path_str
        if "ttnn/operations/" in path_str:
            sub = path_str.split("ttnn/operations/", 1)[1]

        sub_l = sub.lower()
        op_l = op_name.lower()

        # Order of checks defines section ordering
        if (
            "/full/" in sub_l
            or "/full_like/" in sub_l
            or "/rand/" in sub_l
            or "/uniform/" in sub_l
            or "/copy/" in sub_l
        ):
            return "Creation / copy"

        if (
            "/eltwise/unary/" in sub_l
            or "/complex_unary/" in sub_l
            or "/unary/" in sub_l
            and "unary_backward" not in sub_l
        ):
            return "Unary ops"

        if "/eltwise/binary/" in sub_l or "/binary_composite/" in sub_l or "/binary_ng/" in sub_l:
            return "Binary ops"

        if "/eltwise/ternary/" in sub_l:
            return "Ternary ops"

        if "/reduction/" in sub_l or "/prod/" in sub_l or "/topk/" in sub_l or "/accumulation/" in sub_l:
            return "Reductions"

        if (
            "/bernoulli/" in sub_l
            or "/rand/" in sub_l
            or "/uniform/" in sub_l
            or op_l.startswith("ttnn::rand")
            or op_l.startswith("ttnn::uniform")
        ):
            return "Random"

        if "/pool/" in sub_l or "grid_sample" in sub_l or "cdist" in sub_l:
            return "Pooling / distance"

        if "normalization/softmax" in sub_l or "/experimental/dropout/" in sub_l or op_l.endswith("::threshold"):
            return "Softmax / dropout / threshold"

        if (
            "/data_movement/concat/" in sub_l
            or "/data_movement/split/" in sub_l
            or "/data_movement/stack/" in sub_l
            or "/chunk/" in sub_l
        ):
            return "Tensor lists / concat / split"

        if (
            "/data_movement/reshape" in sub_l
            or "/experimental/reshape/" in sub_l
            or "/reshape_view/" in sub_l
            or "/data_movement/view/" in sub_l
            or "/experimental/reshape/view" in sub_l
            or "/transpose/" in sub_l
            or "/permute/" in sub_l
            or "/expand/" in sub_l
            or "/repeat" in sub_l
            or "/roll/" in sub_l
            or "/squeeze/" in sub_l
            or "/unsqueeze/" in sub_l
            or "/tilize" in sub_l
            or "/untilize" in sub_l
            or "/pad/" in sub_l
            or "/view/" in sub_l
        ):
            return "Core tensor ops (shape/view/manipulation)"

        if (
            "/index_fill/" in sub_l
            or "/gather/" in sub_l
            or "/scatter/" in sub_l
            or "/non_zero_indices/" in sub_l
            or "/slice/" in sub_l
            or "/slice_write/" in sub_l
            or "/fill_" in sub_l
            or "/fill/" in sub_l
            or "fill_implicit_tile_padding" in sub_l
            or op_l.startswith("ttnn::fill")
        ):
            return "Indexing / filling"

        if (
            "/core/core.hpp" in sub_l
            or "/typecast/" in sub_l
            or "/quantization/" in sub_l
            or "to_dtype" in op_l
            or "to_layout" in op_l
            or "to_memory_config" in op_l
        ):
            return "Type / quantization / cast"

        if "/matmul/" in sub_l or op_l.endswith("::linear") or "dot" in sub_l:
            return "Matmul / linear / dot"

        if "/conv" in sub_l:
            return "Convolution"

        if "/normalization/" in sub_l:
            return "Normalization"

        if "/transformer/" in sub_l or "/kv_cache/" in sub_l:
            return "Transformer / attention / cache"

        if "/experimental/" in sub_l:
            return "Experimental"

        return "Other"

    # Desired section order
    section_order = [
        "Creation / copy",
        "Unary ops",
        "Binary ops",
        "Ternary ops",
        "Reductions",
        "Random",
        "Pooling / distance",
        "Softmax / dropout / threshold",
        "Tensor lists / concat / split",
        "Core tensor ops (shape/view/manipulation)",
        "Indexing / filling",
        "Type / quantization / cast",
        "Matmul / linear / dot",
        "Convolution",
        "Normalization",
        "Transformer / attention / cache",
        "Experimental",
        "Other",
    ]

    # Build grouped dictionary
    from collections import defaultdict

    grouped = {k: [] for k in section_order}
    for rel, line_no, op_name, var_name in results:
        section = categorize(rel, op_name)
        if section not in grouped:
            grouped["Other"].append((rel, line_no, op_name, var_name))
        else:
            grouped[section].append((rel, line_no, op_name, var_name))

    # Sort entries within each section (by op name then file and line)
    for sec in grouped:
        grouped[sec].sort(key=lambda x: (x[2], x[0], x[1]))

    with OUT_FILE.open("w", encoding="utf-8") as f:
        total = sum(len(v) for v in grouped.values())
        f.write("TTNN operations registered ({} matches)\n\n".format(total))
        for sec in section_order:
            entries = grouped.get(sec, [])
            if not entries:
                continue
            f.write(f"[{sec}] ({len(entries)})\n\n")
            for rel, line_no, op_name, var_name in entries:
                f.write(f"{rel}:{line_no}\n")
                f.write(f'  op: "{op_name}"\n')
                if var_name:
                    f.write(f"  var: {var_name}\n")
                f.write("\n")

    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
