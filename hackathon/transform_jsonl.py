#!/usr/bin/env python3
"""
Transform JSONL file to add constraints based on negative_pocket data.
"""
import json
import sys
from pathlib import Path


def transform_datapoint(datapoint: dict) -> dict:
    """Transform a single datapoint to add constraints from negative_pocket."""
    if "negative_pocket" not in datapoint:
        return datapoint
    
    negative_pocket = datapoint["negative_pocket"]
    
    # Create constraints with negative_pocket format
    constraints = [{
        "negative_pocket": {
            "binder": negative_pocket.get("binder"),
            "contacts": negative_pocket.get("contacts", []),
            "min_distance": negative_pocket.get("min_distance", 10.0),
            "force": True
        }
    }]
    
    # Add constraints to datapoint
    datapoint["constraints"] = constraints
    
    # Remove the original negative_pocket key
    del datapoint["negative_pocket"]
    
    return datapoint


def main():
    if len(sys.argv) != 3:
        print("Usage: python transform_jsonl.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)
    
    # Read, transform, and write
    lines_processed = 0
    with open(output_file, "w") as out_f:
        with open(input_file, "r") as in_f:
            for line_num, line in enumerate(in_f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    datapoint = json.loads(line)
                    transformed = transform_datapoint(datapoint)
                    out_f.write(json.dumps(transformed) + "\n")
                    lines_processed += 1
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    sys.exit(1)
    
    print(f"Successfully transformed {lines_processed} datapoints")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    main()

