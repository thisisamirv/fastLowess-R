#!/usr/bin/env python3
import json
import os
import shutil
import sys

def clean_checksums(vendor_dir):
    print(f"Cleaning checksums in {vendor_dir}...")
    
    # 1. Remove non-essential directories
    STRIP_DIRS = ["tests", "benches", "examples", "doc", "docs", ".github", ".config"]
    
    for root, dirs, files in os.walk(vendor_dir):
        # Remove hidden directories and STRIP_DIRS
        for d in list(dirs):
            if d.startswith(".") or d in STRIP_DIRS:
                full_path = os.path.join(root, d)
                print(f"  Stripping directory: {full_path}")
                shutil.rmtree(full_path, ignore_errors=True)
                dirs.remove(d)

    # 2. Remove hidden files (except checksums)
    for root, dirs, files in os.walk(vendor_dir):
        for f in files:
            if f.startswith(".") and f != ".cargo-checksum.json":
                full_path = os.path.join(root, f)
                print(f"  Removing hidden file: {full_path}")
                try:
                    os.remove(full_path)
                except:
                    pass

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
                    # 1. Were hidden files (handled above)
                    # 2. Don't exist on disk (were inside stripped dirs)
                    new_files = {}
                    for k, v in original_files.items():
                        # Normalized path for checking existence
                        exists = os.path.exists(os.path.join(root, k))
                        is_hidden = any(part.startswith(".") for part in k.split("/"))
                        
                        if exists and not is_hidden:
                            new_files[k] = v
                        # If it doesn't exist or is hidden, it is excluded from checksum
                    
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
