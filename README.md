# SmBoP_orig_cbr

Choosing 5 out of 10 random cases:
Config: orig_cbr_rand_data_loader_config.jsonnet
Data Loader: cbr_with_same_schema_cat_rand_cases.py
Dataset Reader: spider_basic_pkl.py

Using a file with fixed cases:
Config: orig_cbr_config.jsonnet
Data Loader: default
Dataset Reader: spider_round_robin_cases.py

Models: 70% using cases, 30% vanilla training. Cases used only in question embedder
smbop_with_concat_cases_vanilla_mix.py: use while training
smbop_with_concat_cases_vanilla_mix_new_mets.py: use while doing eval as it has support for top k mwetrics.

Note: this was done so we always use cases ( no random toss) while evaluating the model
