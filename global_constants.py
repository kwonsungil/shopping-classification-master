import os

base_dir = "D:\\kown\\kao"
base_data_dir = os.path.join(base_dir, "data")

raw_dataset = os.path.join(base_data_dir, "original")
image_dataset = os.path.join(base_data_dir, "image")
prerocess_dataset = os.path.join(base_data_dir, "preprocess")
escaped_dataset = os.path.join(base_data_dir, "escaped")

classification_training_dataset_dir = os.path.join(base_data_dir, "classification_training_dataset")
classification_dev_dataset_dir = os.path.join(base_data_dir, "classification_dev_dataset")

base_model_dir = os.path.join(base_dir, "logs")

map_cate_file = os.path.join(base_data_dir, "label.json")
vocap_char_file = os.path.join(base_data_dir, 'vocab_char.txt')


