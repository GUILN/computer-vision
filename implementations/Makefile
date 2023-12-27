MAIN_FOLDER = pix2pix_lung_cancer
DATA_FOLDER = ./data
PATCHES_FOLDER = ${DATA_FOLDER}/LUNA_patches
INPUT_DATA_FOLDER = ${DATA_FOLDER}/saved_input_data

# Data prep steps:
# 1. Extract
# 2. Generate

# CI
ci/format:
	@echo "Formatting code..."
	black --ipynb --exclude='.py' ./pix2pix_lung_cancer \
	&& black --exclude='.ipynb' ./pix2pix_lung_cancer

# Data prep
data/extract:
	@echo "Extracting data..."
	cd ./${MAIN_FOLDER} && ./extract_all.sh

data/generate:
	@echo "Generating data..."
	cd ./${MAIN_FOLDER} && python ./generate_test_data_main.py \
		--patches_dir $(PATCHES_FOLDER) \
		--save_dir $(INPUT_DATA_FOLDER)

# Training
train:
	@echo "Training model..."
	cd ./$(MAIN_FOLDER) \
	&& python ./train_main.py \
		--input_data_dir $(INPUT_DATA_FOLDER) 
