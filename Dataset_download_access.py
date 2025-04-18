import gdown
import zipfile
import os
import glob

#CONFIG

file_id = '1kAIW8Zh4irapOLh8LJaoVpWhWI29vorg'  
zip_name = 'Dataset.zip'
extracted_folder = 'Dataset'

# STEP 1: DOWNLOAD FROM GOOGLE DRIVE
if not os.path.exists(zip_name):
    print("Downloading the dataset from Google Drive...")
    gdown.download(f'https://drive.google.com/uc?id={file_id}', zip_name, quiet=False)
else:
    print("Zip file already exists. Skipping download.")

# STEP 2: EXTRACT ZIP FILE
if not os.path.exists(extracted_folder):
    print("Extracting the dataset...")
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall()
else:
    print("Dataset already extracted.")

# STEP 3: ACCESS IMAGES
paths = {
    "train_cover": glob.glob(f"{extracted_folder}/train/cover/*"),
    "train_mark": glob.glob(f"{extracted_folder}/train/mark/*"),
    "test_cover": glob.glob(f"{extracted_folder}/test/cover/*"),
    "test_mark": glob.glob(f"{extracted_folder}/test/mark/*"),
}

# Optional: Check and print counts
for name, files in paths.items():
    print(f"{name}: {len(files)} images")

# Sample preview
print("\nSample from train/cover:")
for img in paths["train_cover"][:5]:
    print(img)

