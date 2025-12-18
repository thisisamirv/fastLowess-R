#!/bin/bash
set -e

echo "📦 Preparing package for CRAN submission..."

# 1. Vendor Dependencies
echo "   -> Vendoring Rust dependencies..."
mkdir -p .cargo
cargo vendor > .cargo/config.toml

# 2. Fix Checksums (remove files that R CMD build excludes, like .gitignore)
echo "   -> Fixing checksums for R compatibility..."
python3 -c '
import os, json

VENDOR_DIR = "vendor"
for root, _, files in os.walk(VENDOR_DIR):
    if ".cargo-checksum.json" in files:
        path = os.path.join(root, ".cargo-checksum.json")
        with open(path, "r") as f: d = json.load(f)
        
        files_dict = d.get("files", {})
        removals = []
        
        # files to remove from checksums because R excludes them from the tarball
        for k in files_dict:
            if k.endswith(".gitignore") or "/.github" in k:
                removals.append(k)
                # also remove from disk to be sure
                full_path = os.path.join(root, k)
                if os.path.exists(full_path):
                    os.remove(full_path)
        
        if removals:
            print(f"      Fixed {len(removals)} entries in {path}")
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
