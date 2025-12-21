#!/usr/bin/env python3
import json
import os
import sys

def clean_checksums(vendor_dir):
    print(f"Cleaning checksums in {vendor_dir}...")
    updated_count = 0
    for root, dirs, files in os.walk(vendor_dir):
        if ".cargo-checksum.json" in files:
            filepath = os.path.join(root, ".cargo-checksum.json")
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                if "files" in data:
                    original_files = data["files"]
                    # Remove any keys that start with .git (like .gitignore)
                    # or are generally excluded by R CMD build
                    new_files = {
                        k: v for k, v in original_files.items() 
                        if not any(part.startswith(".git") for part in k.split("/"))
                    }
                    
                    if len(original_files) != len(new_files):
                        print(f"  Updated {filepath}: removed {len(original_files) - len(new_files)} files")
                        data["files"] = new_files
                        with open(filepath, "w") as f:
                            json.dump(data, f)
                        updated_count += 1
            except Exception as e:
                print(f"  Error processing {filepath}: {e}")
    
    print(f"Done. Updated {updated_count} checksum files.")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "src/vendor"
    clean_checksums(target)
