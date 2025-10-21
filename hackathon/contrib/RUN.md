# How to run

conda activate boltz-LABind-feature-pocket


Download the checkpoints

python hackathon/predict_hackathon.py \
    --input-jsonl hackathon_data/datasets/asos_public/asos_public.jsonl \
    --msa-dir hackathon_data/datasets/asos_public/msa/ \
    --submission-dir /home/ubuntu/templates/repo/boltz-hackathon-template/output/submission_dir \
    --intermediate-dir hackathon_data/intermediate_files/asos_public \
    --result-folder /home/ubuntu/templates/repo/boltz-hackathon-template/output
