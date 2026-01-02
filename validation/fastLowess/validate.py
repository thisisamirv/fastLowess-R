import json
import os
from pathlib import Path
import numpy as np
import fastlowess

def process_file(input_path: Path, output_dir: Path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Extract params
    params = data.get("params", {})
    input_data = data.get("input", {})
    x = np.array(input_data.get("x", []), dtype=float)
    y = np.array(input_data.get("y", []), dtype=float)
    
    fraction = params.get("fraction")
    iterations = params.get("iterations")
    delta = params.get("delta")
    
    # Configure arguments for fastlowess.smooth
    kwargs = {
        "fraction": fraction,
        "iterations": iterations,
        "scaling_method": "mar",     # Matching Rust validate.rs: .scaling_method(MAR)
        "boundary_policy": "noboundary", # Matching Rust validate.rs: .boundary_policy(NoBoundary)
        "parallel": True             # Matching Rust validate.rs: .parallel(true)
    }
    
    # fastlowess.smooth handles delta=None internally (defaults to 0.0), 
    # but let's pass it explicitly if it exists in the JSON params
    if delta is not None:
        kwargs["delta"] = delta
        
    # Run smoothing
    try:
        # returns fitted_y array
        fitted = fastlowess.smooth(x, y, **kwargs)
        
        # Update result in data structure
        if "result" not in data:
            data["result"] = {}
        data["result"]["fitted"] = fitted.y.tolist()
        
        # Write output
        output_path = output_dir / input_path.name
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        print(f"Failed to process {input_path.name}: {e}")

def main():
    # Relative paths matching the Rust implementation's directory structure expectations
    # Rust path: "../output/r" from "validation/fastLowess/src/" matches 
    # Python path: "../output/r" from "validation/fastLowess/"
    
    script_dir = Path(__file__).parent.resolve()
    input_dir = script_dir.parent / "output" / "r"
    output_dir = script_dir.parent / "output" / "fastLowess"
    
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist. Run validate.R first.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    entries = sorted([p for p in input_dir.iterdir() if p.suffix == ".json"])
    
    for entry in entries:
        print(f"Processing {entry.name}")
        process_file(entry, output_dir)

if __name__ == "__main__":
    main()
