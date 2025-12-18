#!/bin/bash
set -e

echo "📦 Preparing package for CRAN submission..."

# 1. Vendor Dependencies
echo "   -> Vendoring Rust dependencies..."
mkdir -p .cargo
cargo vendor > .cargo/config.toml

# 2. Clean hidden files and fix checksums
echo "   -> Cleaning hidden files from vendor..."
python3 -c '
import os, json, shutil, re

VENDOR_DIR = "vendor"

# Pattern to match hidden files anywhere in path (e.g., "tests/fst/.gitignore")
HIDDEN_PATTERN = re.compile(r"(^|/)\.")

# Remove hidden directories
for root, dirs, files in os.walk(VENDOR_DIR, topdown=True):
    for d in list(dirs):
        if d.startswith("."):
            full_path = os.path.join(root, d)
            print(f"      Removing dir: {full_path}")
            shutil.rmtree(full_path)
            dirs.remove(d)

# Remove ALL hidden files from disk (at any depth)
for root, dirs, files in os.walk(VENDOR_DIR):
    for f in files:
        if f.startswith(".") and f != ".cargo-checksum.json":
            full_path = os.path.join(root, f)
            print(f"      Removing file: {full_path}")
            os.remove(full_path)

# Fix checksums - remove entries for ANY hidden file in the path
for root, dirs, files in os.walk(VENDOR_DIR):
    if ".cargo-checksum.json" in files:
        path = os.path.join(root, ".cargo-checksum.json")
        with open(path, "r") as f: d = json.load(f)
        
        files_dict = d.get("files", {})
        # Match: starts with ".", OR contains "/." anywhere (hidden file in subdir)
        removals = [k for k in files_dict if HIDDEN_PATTERN.search(k)]
        
        if removals:
            print(f"      Fixed {len(removals)} checksum entries in {path}")
            for k in removals: del files_dict[k]
            d["files"] = files_dict
            with open(path, "w") as f: json.dump(d, f)
'

# 3. Generate AUTHORS file
echo "   -> Generating inst/AUTHORS..."
mkdir -p inst
cargo metadata --format-version 1 > cargo_metadata_temp.json

python3 -c '
import json
with open("cargo_metadata_temp.json") as f: data = json.load(f)

seen = set()
with open("inst/AUTHORS", "w") as f:
    f.write("Authors and Copyright Holders for Rust Dependencies:\n\n")
    for pkg in data["packages"]:
        name = pkg["name"]
        if name == "fastLowess": continue 
        
        # Deduplicate
        if name in seen: continue
        seen.add(name)
        
        version = pkg["version"]
        authors = ", ".join(pkg["authors"])
        license = pkg.get("license", "Unknown")
        
        f.write(f"Package: {name} ({version})\n")
        f.write(f"Authors: {authors}\n")
        f.write(f"License: {license}\n")
        f.write("-" * 40 + "\n")
'
rm cargo_metadata_temp.json

echo "✅ Preparation complete!"
echo "   1. Dependencies are in 'vendor/'"
echo "   2. Local config is in '.cargo/config.toml'"
echo "   3. Attribution is in 'inst/AUTHORS'"
echo ""
echo "You can now run: make install"
