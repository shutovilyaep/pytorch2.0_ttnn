#!/usr/bin/env python3
import json
import os
import sys


def rewrite_path(p: str, src: str, dst: str) -> str:
    if not p:
        return p
    if p.startswith(src):
        return dst + p[len(src) :]
    return p


def main():
    if len(sys.argv) < 5:
        print(
            "Usage: rewrite_compile_commands.py <in> <out> <from_prefix> <to_prefix>",
            file=sys.stderr,
        )
        sys.exit(2)

    src_path = sys.argv[1]
    dst_path = sys.argv[2]
    from_prefix = sys.argv[3].rstrip("/")
    to_prefix = sys.argv[4].rstrip("/")

    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        # Required fields
        entry["file"] = rewrite_path(entry.get("file", ""), from_prefix, to_prefix)
        entry["directory"] = rewrite_path(entry.get("directory", ""), from_prefix, to_prefix)
        # Optional fields
        if "command" in entry and isinstance(entry["command"], str):
            entry["command"] = entry["command"].replace(from_prefix, to_prefix)
        if "arguments" in entry and isinstance(entry["arguments"], list):
            entry["arguments"] = [
                (arg.replace(from_prefix, to_prefix) if isinstance(arg, str) else arg) for arg in entry["arguments"]
            ]

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
