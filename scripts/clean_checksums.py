#!/usr/bin/env python3
import json
import os
import shutil
import sys

def clean_checksums(vendor_dir):
    print(f"Cleaning checksums in {vendor_dir}...")
    
    # 1. Remove non-essential directories to save space and avoid R CMD build issues
    # CRAN has a 5MB limit, and R CMD build sometimes excludes 'tests' directories.
    STRIP_DIRS = ["tests", "benches", "examples", "doc", "docs"]
    
    for root, dirs, files in os.walk(vendor_dir):
        for d in list(dirs):
            if d in STRIP_DIRS:
                full_path = os.path.join(root, d)
                print(f"  Stripping directory: {full_path}")
                shutil.rmtree(full_path)
                dirs.remove(d)

    updated_count = 0
    for root, dirs, files in os.walk(vendor_dir):
        if ".cargo-checksum.json" in files:
            filepath = os.path.join(root, ".cargo-checksum.json")
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                if "files" in data:
                    original_files = data["files"]
                    
                    # Remove keys that:
                    # 1. Start with .git (like .gitignore)
                    # 2. Don't exist on disk (were stripped or excluded by R CMD build)
                    new_files = {}
                    for k, v in original_files.items():
                        is_hidden = any(part.startswith(".git") for part in k.split("/"))
                        exists = os.path.exists(os.path.join(root, k))
                        
                        if not is_hidden and exists:
                            new_files[k] = v
                        elif not exists:
                            # Silently remove missing files from checksum to satisfy cargo
                            pass
                        else:
                            print(f"  Removing checksum entry for: {k}")
                    
                    if len(original_files) != len(new_files):
                        print(f"  Updated {filepath}: removed {len(original_files) - len(new_files)} entries")
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
