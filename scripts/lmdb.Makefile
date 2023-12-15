# local Mac
DATA_ROOT_DIR = /Users/king/Documents/beiye-Ein/语料/ocr
# gpu
#DATA_ROOT_DIR = /data/jinlong/ocr_data

prepare-data:
	python scripts/process_lmdb.py dump-lmdb -i $(DATA_ROOT_DIR)/benchmark_dataset/scene/scene_train -o $(DATA_ROOT_DIR)/benchmark_dataset/scene-new/scene_train
	python scripts/process_lmdb.py dump-lmdb -i $(DATA_ROOT_DIR)/benchmark_dataset/scene/scene_val -o $(DATA_ROOT_DIR)/benchmark_dataset/scene-new/scene_val
	python scripts/process_lmdb.py dump-lmdb -i $(DATA_ROOT_DIR)/benchmark_dataset/scene/scene_test -o $(DATA_ROOT_DIR)/benchmark_dataset/scene-new/scene_test
	python scripts/process_lmdb.py dump-lmdb -i $(DATA_ROOT_DIR)/benchmark_dataset/web/web_train -o $(DATA_ROOT_DIR)/benchmark_dataset/web-new/web_train
	python scripts/process_lmdb.py dump-lmdb -i $(DATA_ROOT_DIR)/benchmark_dataset/web/web_val -o $(DATA_ROOT_DIR)/benchmark_dataset/web-new/web_val
	python scripts/process_lmdb.py dump-lmdb -i $(DATA_ROOT_DIR)/benchmark_dataset/web/web_test -o $(DATA_ROOT_DIR)/benchmark_dataset/web-new/web_test
