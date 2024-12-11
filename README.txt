To replicate the results:
1. Create a folder named "data_raw".
2. Download the two datasets (both found on https://snap.stanford.edu) and extract them into folders named "deezer_europe" and "reddit_hyperlink".
3. Run "proprocess_deezer_data.py" and "preprocess_reddit_data.py" to preprocess the datasets. This will create a folder named "data" with processed datasets stored as Python Pickle files.
4. Run "main.py" (Usage: python main.py <dataset> <model>). The results and model checkpoint will be automatically recorded. Tensorboard is used to create training logs.
5. Run statistical tests with "paired_t_tests.py".