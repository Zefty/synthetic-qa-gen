python SyntheticQAGenerator.py \
    --csv_path 'splitted_covid_dump-covidQA.csv' \
    --shard true \
    --start_shard 0 \
    --end_shard 20 \
    --num_shards 20 \
    --chunk_size 10
