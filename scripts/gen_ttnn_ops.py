#!/usr/bin/env python3
import os
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIR = REPO_ROOT / "torch_ttnn/cpp_extension/third-party/tt-metal/ttnn/cpp/ttnn/operations"
OUT_FILE = REPO_ROOT / "ttnn_ops_implemented.txt"
OUT_GROUPED_FILE = REPO_ROOT / "ttnn_ops_grouped.txt"
OPEN_REGISTRATION_CPP = REPO_ROOT / "torch_ttnn/cpp_extension/ttnn_cpp_extension/src/open_registration_extension.cpp"


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


def find_unary_macro_registrations(text: str):
    # Capture macro usages like: REGISTER_UNARY_OPERATION(acos, ACOS)
    # and variations: REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(name, TYPE)
    # We only need the first argument (the variable/op name)
    pattern = re.compile(
        r'REGISTER_UNARY_OPERATION(?:_[A-Z_]+)?\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,',
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

    # Sort by op name then file
    results.sort(key=lambda x: (x[2], x[0], x[1]))

    # Write original flat list for backward compatibility

    with OUT_FILE.open("w", encoding="utf-8") as f:
        f.write("TTNN operations registered ({} matches)\n\n".format(len(results)))
        for rel, line_no, op_name, var_name in results:
            f.write(f"{rel}:{line_no}\n")
            f.write(f"  op: \"{op_name}\"\n")
            if var_name:
                f.write(f"  var: {var_name}\n")
            f.write("\n")

    print(f"Wrote {OUT_FILE}")

    # Build a set of op base names implemented in open_registration_extension.cpp
    implemented_ops = set()
    try:
        cpp_text = OPEN_REGISTRATION_CPP.read_text(encoding="utf-8", errors="ignore")
        # Capture m.impl("name" ...)
        for m in re.finditer(r'm\.impl\(\s*"([^"]+)"', cpp_text):
            name = m.group(1)
            # Ignore fully-qualified names like aten::*
            if "::" in name:
                continue
            base = name.split(".", 1)[0]
            if base:
                implemented_ops.add(base)
    except Exception:
        # If not found, leave the set empty
        implemented_ops = set()

    # Group TTNN ops similar to groups in open_registration_extension.cpp
    def group_for(rel_path: str, op_name: str) -> str:
        p = rel_path.replace("\\", "/")
        if "/operations/" in p:
            after = p.split("/operations/", 1)[1]
        else:
            after = p
        parts = after.split("/")
        # Heuristic grouping
        if len(parts) >= 2 and parts[0] == "eltwise":
            if parts[1].startswith("unary") or parts[1] in {"complex_unary", "unary_backward"}:
                return "Unary ops"
            if parts[1].startswith("binary") or parts[1] in {"binary_composite", "binary_backward"}:
                return "Binary ops"
            if parts[1].startswith("ternary") or parts[1] in {"ternary_backward"}:
                return "Binary/Ternary ops"
        if parts[0] == "reduction":
            return "Reductions"
        if parts[0] in {"bernoulli", "rand", "uniform", "random"}:
            return "Random ops"
        if parts[0] in {"data_movement", "copy", "typecast"}:
            return "Core tensor ops"
        if parts[0] == "core":
            return "Core ops"
        if parts[0] == "normalization":
            return "Normalization ops"
        if parts[0] in {"matmul", "conv", "pool"}:
            return "Linear/Conv/Pool ops"
        if parts[0] in {"transformer"}:
            return "Transformer ops"
        if parts[0] in {"kv_cache"}:
            return "KV cache ops"
        if parts[0] in {"experimental"}:
            return "Experimental ops"
        return "Other ops"

    def ttnn_base_name(op_name: str) -> str:
        # Strip namespaces like ttnn::, ttnn::experimental::, ttnn::prim::
        if "::" in op_name:
            return op_name.split("::")[-1]
        return op_name

    groups = {}
    implemented_count = 0
    for rel, line_no, op_name, var_name in results:
        g = group_for(rel, op_name)
        base = ttnn_base_name(op_name)
        is_impl = base in implemented_ops
        if is_impl:
            implemented_count += 1
        groups.setdefault(g, []).append({
            "rel": rel,
            "line": line_no,
            "op": op_name,
            "var": var_name,
            "base": base,
            "implemented": is_impl,
        })

    # Sort groups and entries
    for g, entries in groups.items():
        entries.sort(key=lambda e: (e["op"], e["rel"], e["line"]))

    ordered_group_names = [
        "Core ops",
        "Unary ops",
        "Binary ops",
        "Binary/Ternary ops",
        "Reductions",
        "Random ops",
        "Core tensor ops",
        "Normalization ops",
        "Linear/Conv/Pool ops",
        "Transformer ops",
        "KV cache ops",
        "Experimental ops",
        "Other ops",
    ]

    with OUT_GROUPED_FILE.open("w", encoding="utf-8") as f:
        total = len(results)
        f.write(f"TTNN operations grouped report\n")
        f.write(f"Total ops discovered: {total}\n")
        f.write(f"Implemented via open_registration_extension.cpp: {implemented_count}\n")
        f.write("\n")

        for group_name in ordered_group_names:
            entries = groups.get(group_name, [])
            if not entries:
                continue
            impl_in_group = sum(1 for e in entries if e["implemented"]) 
            f.write(f"### {group_name} ({impl_in_group}/{len(entries)})\n")
            for e in entries:
                prefix = "" if e["implemented"] else "# "
                status = "OK" if e["implemented"] else "TODO"
                var_part = f" var:{e['var']}" if e["var"] else ""
                f.write(f"{prefix}{e['op']}  -- {status}  [{e['rel']}:{e['line']}{var_part}]\n")
            f.write("\n")

    print(f"Wrote {OUT_GROUPED_FILE}")


if __name__ == "__main__":
    main()


