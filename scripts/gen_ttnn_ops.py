#!/usr/bin/env python3
import os
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIR = REPO_ROOT / "torch_ttnn/cpp_extension/third-party/tt-metal/ttnn/cpp/ttnn/operations"
OUT_FILE = REPO_ROOT / "ttnn_ops_implemented.txt"
OUT_GROUPED_FILE = REPO_ROOT / "ttnn_ops_grouped.txt"
OPEN_REGISTRATION_CPP = REPO_ROOT / "torch_ttnn/cpp_extension/ttnn_cpp_extension/src/open_registration_extension.cpp"
OUT_PYTORCH_OPS_FILE = REPO_ROOT / "pytorch_ops_scanned.txt"
OUT_PYTORCH_MISSING_FILE = REPO_ROOT / "pytorch_missing_in_open_registration.txt"


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

    # Build a set of ttnn op base names used via wrappers/aliases in open_registration_extension.cpp
    def extract_ttnn_bases_from_text(text: str) -> set[str]:
        bases: set[str] = set()

        # 1) Collect alias mappings: using Alias = tt_eager::ext::Wrapper<ttnn::op[, ttnn::op2]>;
        alias_to_bases: dict[str, set[str]] = {}
        using_rx = re.compile(
            r"using\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*tt_eager::ext::[A-Za-z_][A-Za-z0-9_]*\s*<([^;>]+)>\s*;",
            re.MULTILINE | re.DOTALL,
        )
        for um in using_rx.finditer(text):
            alias = um.group(1)
            tpl = um.group(2)
            tt_ops = set(re.findall(r"ttnn::[A-Za-z0-9_:]+", tpl))
            alias_to_bases[alias] = {op.split("::")[-1] for op in tt_ops}

        # 2) Parse inline wrappers in TORCH_FN(... tt_eager::ext::Wrapper<ttnn::...> ::invoke ...)
        torchfn_rx = re.compile(r"TORCH_FN\(\s*([^\)]+)\)")
        for tf in torchfn_rx.finditer(text):
            inside = tf.group(1)
            # Case A: Alias::invoke
            m_alias = re.search(r"([A-Za-z_][A-Za-z0-9_]*)::invoke", inside)
            if m_alias:
                alias = m_alias.group(1)
                if alias in alias_to_bases:
                    bases.update(alias_to_bases[alias])
            # Case B: tt_eager::ext::Wrapper<...>::invoke
            m_tpl = re.search(r"tt_eager::ext::[A-Za-z_][A-Za-z0-9_]*\s*<([^>]+)>\s*::invoke", inside)
            if m_tpl:
                tpl = m_tpl.group(1)
                tt_ops = set(re.findall(r"ttnn::[A-Za-z0-9_:]+", tpl))
                bases.update(op.split("::")[-1] for op in tt_ops)
            # Case C: random_like_rand special-case (no template args)
            if "tt_eager::ext::random_like_rand::" in inside:
                # random_like_rand uses rand + (floor + typecast for casting)
                bases.update({"rand", "floor", "typecast"})

        # 3) Some wrappers may specify Complex ops returning ComplexTensor (handled as ttnn::angle/conj etc.)
        # Already covered by template parsing above.
        return bases

    implemented_ttnn_ops = set()
    registered_aten_bases = set()
    try:
        cpp_text = OPEN_REGISTRATION_CPP.read_text(encoding="utf-8", errors="ignore")
        implemented_ttnn_ops = extract_ttnn_bases_from_text(cpp_text)
        # Also record registered aten base names for separate PyTorch ops report
        for m in re.finditer(r'm\.impl\(\s*"([^"]+)"', cpp_text):
            name = m.group(1)
            if "::" in name:
                continue
            base = name.split(".", 1)[0]
            if base:
                registered_aten_bases.add(base)
    except Exception:
        implemented_ttnn_ops = set()
        registered_aten_bases = set()

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
        is_impl = base in implemented_ttnn_ops
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

    # =========================
    # Extended scan of PyTorch ops (allowlist + native + named registrations)
    # =========================

    def discover_candidate_roots() -> list[Path]:
        roots: list[Path] = []
        env = os.environ.get("PYTORCH_SRC_ROOT")
        if env:
            p = Path(env)
            if p.exists():
                roots.append(p)
        # common vendor locations relative to this repo
        for rel in ["third_party/pytorch", "../pytorch", "pytorch", ".."]:
            try:
                p = (REPO_ROOT / rel).resolve()
            except Exception:
                continue
            if p.exists() and p.is_dir():
                roots.append(p)
        # also consider REPO_ROOT itself as fallback
        roots.append(REPO_ROOT)
        # de-dup
        uniq = []
        seen = set()
        for r in roots:
            if str(r) not in seen:
                uniq.append(r)
                seen.add(str(r))
        return uniq

    def find_first_file(roots: list[Path], filename: str) -> list[Path]:
        found: list[Path] = []
        for root in roots:
            for dirpath, _, filenames in os.walk(root):
                if filename in filenames:
                    found.append(Path(dirpath) / filename)
        return found

    def collect_ops_from_allowlist(path: Path) -> set[str]:
        ops: set[str] = set()
        try:
            import json
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            # recursively gather all string values
            def gather(x):
                if isinstance(x, str):
                    ops.add(x.split(".", 1)[0])
                elif isinstance(x, dict):
                    for v in x.values():
                        gather(v)
                elif isinstance(x, list):
                    for v in x:
                        gather(v)
            gather(data)
        except Exception:
            pass
        return ops

    def collect_ops_from_native_functions(path: Path) -> set[str]:
        ops: set[str] = set()
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for m in re.finditer(r"^\s*-\s*func:\s*([^\(\s]+)\(", text, re.MULTILINE):
                name = m.group(1)
                if "::" in name:
                    name = name.split("::")[-1]
                base = name.split(".", 1)[0]
                ops.add(base)
        except Exception:
            pass
        return ops

    def collect_ops_from_named_registrations(path: Path) -> set[str]:
        ops: set[str] = set()
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for m in re.finditer(r"m\\.impl\\(\\s*\"([^\"]+)\"", text):
                nm = m.group(1)
                if "::" in nm:
                    continue
                base = nm.split(".", 1)[0]
                ops.add(base)
        except Exception:
            pass
        return ops

    roots = discover_candidate_roots()
    allowlist_paths = find_first_file(roots, "allowlist_for_publicAPI.json")
    native_yaml_paths = find_first_file(roots, "native_functions.yaml")
    named_regs_paths = find_first_file(roots, "NamedRegistrations.cpp")

    pytorch_ops: dict[str, set[str]] = {"allowlist": set(), "native": set(), "named_regs": set()}
    for p in allowlist_paths:
        pytorch_ops["allowlist"].update(collect_ops_from_allowlist(p))
    for p in native_yaml_paths:
        pytorch_ops["native"].update(collect_ops_from_native_functions(p))
    for p in named_regs_paths:
        pytorch_ops["named_regs"].update(collect_ops_from_named_registrations(p))

    union_ops = set().union(*pytorch_ops.values())

    try:
        with OUT_PYTORCH_OPS_FILE.open("w", encoding="utf-8") as f:
            f.write("PyTorch ops scan report\n")
            f.write(f"Roots searched: {', '.join(str(r) for r in roots)}\n")
            f.write(f"allowlist files: {len(allowlist_paths)}; native_functions.yaml files: {len(native_yaml_paths)}; NamedRegistrations.cpp files: {len(named_regs_paths)}\n")
            f.write(f"Total unique ops discovered: {len(union_ops)}\n")
            f.write(f"From allowlist: {len(pytorch_ops['allowlist'])}\n")
            f.write(f"From native_functions: {len(pytorch_ops['native'])}\n")
            f.write(f"From NamedRegistrations: {len(pytorch_ops['named_regs'])}\n\n")

            for source, sops in pytorch_ops.items():
                if not sops:
                    continue
                f.write(f"### {source} ({len(sops)})\n")
                for op in sorted(sops):
                    f.write(f"{op}\n")
                f.write("\n")
        print(f"Wrote {OUT_PYTORCH_OPS_FILE}")
    except Exception:
        pass

    # Report PyTorch ops missing in our open_registration (by base name)
    try:
        missing = sorted(op for op in union_ops if op not in registered_aten_bases)
        with OUT_PYTORCH_MISSING_FILE.open("w", encoding="utf-8") as f:
            f.write("PyTorch ops NOT registered in open_registration_extension.cpp (base-name comparison)\n")
            f.write(f"Total missing: {len(missing)}\n\n")
            for op in missing:
                f.write(f"{op}\n")
        print(f"Wrote {OUT_PYTORCH_MISSING_FILE}")
    except Exception:
        pass


if __name__ == "__main__":
    main()


