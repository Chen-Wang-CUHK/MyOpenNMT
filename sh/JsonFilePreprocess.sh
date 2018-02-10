# Generate .txt data file from .json file both w/ and w/o rich features
nohup python -u JsonFilePreprocess.py \
-data_dir ../data/corenlp_data \
-stop_words_file ../data/corenlp_data/stfd_stopwords/corenlp_stopwords.json \
> JsonFilePreprocess_log.out &