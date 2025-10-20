#!/usr/bin/env python3
"""Test script to verify YAML generation with proper constraint formatting"""
import yaml
import sys
from pathlib import Path

# Add hackathon to the path
sys.path.insert(0, str(Path(__file__).parent / "hackathon"))

from hackathon_api import Datapoint

# Read the first line of the JSONL
with open('hackathon_data/datasets/asos_public/asos_public_test1.jsonl', 'r') as f:
    line = f.readline()
    
datapoint = Datapoint.from_json(line)

# Build the test dict
doc = {
    "version": 1,
    "sequences": [
        {
            "protein": {
                "id": datapoint.proteins[0].id,
                "sequence": datapoint.proteins[0].sequence[:50] + "...",  # Truncate for readability
                "msa": datapoint.proteins[0].msa
            }
        },
        {
            "ligand": {
                "id": datapoint.ligands[0].id,
                "smiles": datapoint.ligands[0].smiles
            }
        }
    ]
}

if datapoint.constraints:
    doc["constraints"] = datapoint.constraints

# Custom YAML formatting to keep contacts as inline lists
class FlowListDumper(yaml.SafeDumper):
    pass

def represent_list(dumper, data):
    # If this is a constraint contacts list (list of 2-element lists), use flow style
    if data and isinstance(data, list) and len(data) > 0:
        if all(isinstance(item, list) and len(item) == 2 for item in data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

FlowListDumper.add_representer(list, represent_list)

print('Generated YAML with corrected constraint format:')
print('=' * 80)
yaml_output = yaml.dump(doc, Dumper=FlowListDumper, sort_keys=False, default_flow_style=False)
print(yaml_output)
print('=' * 80)

