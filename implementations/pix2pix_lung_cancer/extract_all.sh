#!/bin/bash

# Extract all the files
echo "Extracting all the files..."
for i in {2..9}
do
    echo "Extracting subset$i..."
    python ./preprocess/main.py --data_dir ./data/LUNA16/subset$i \
     --patches_dir ./data/LUNA_patches --annotations_file ./data/LUNA16/annotations.csv \
      --candidates_file ./data/LUNA16/candidates.csv
    echo "Done subset$i!"
done

echo "Done!"