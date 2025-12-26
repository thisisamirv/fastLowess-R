#!/bin/bash
set -e

echo "📦 Preparing package for CRAN submission..."

# 1. Vendor Dependencies
echo "   -> Vendoring Rust dependencies..."
mkdir -p src/cargo
(cd src && cargo vendor vendor > cargo/config.toml)

# 2. Clean vendor: strip tests/benches/examples, fix checksums
echo "   -> Cleaning vendor directory and fixing checksums..."
python3 scripts/clean_checksums.py src/vendor

# 3. Generate AUTHORS file
echo "   -> Generating inst/AUTHORS..."
mkdir -p inst
(cd src && cargo metadata --format-version 1 > ../cargo_metadata_temp.json)

python3 -c '
import json
with open("cargo_metadata_temp.json") as f: data = json.load(f)

seen = set()
with open("inst/AUTHORS", "w") as f:
    f.write("Authors and Copyright Holders for Rust Dependencies:\n\n")
    for pkg in data["packages"]:
        name = pkg["name"]
        if name == "fastLowess-R": continue 
        
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
echo "   1. Dependencies are in 'src/vendor/'"
echo "   2. Local config is in 'src/cargo/config.toml'"
echo "   3. Attribution is in 'inst/AUTHORS'"
echo ""
echo "You can now run: make install"
