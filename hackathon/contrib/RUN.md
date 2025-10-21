conda activate boltz-LABind-feature-pocket

python hackathon/predict_hackathon.py \
    --input-jsonl hackathon_data/datasets/asos_public/asos_public.jsonl \
    --msa-dir hackathon_data/datasets/asos_public/msa/ \
    --submission-dir /home/ubuntu/templates/repo/boltz-hackathon-template/output/submission_dir \
    --intermediate-dir hackathon_data/intermediate_files/asos_public \
    --result-folder /home/ubuntu/templates/repo/boltz-hackathon-template/output  \
    --use-auto-pocket-scanner


python predict_binding_sites.py -op example/out/ -if example/protein.fa -il example/smiles.txt