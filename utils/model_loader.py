import os
import gdown
import zipfile

def download_and_extract_models_from_drive(file_id, output_zip='models.zip', extract_to='Models'):
    # Only download if not already extracted
    if not os.path.exists(extract_to):
        os.makedirs(extract_to, exist_ok=True)

        # Step 1: Download from Google Drive
        url = f"https://drive.google.com/uc?id={file_id}"
        print("Downloading model zip from Google Drive...")
        gdown.download(url, output_zip, quiet=False)

        # Step 2: Extract
        print("📦 Extracting models...")
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        print(f"✅ Models extracted to: {extract_to}")
    else:
        print(f"✅ Models already available at: {extract_to}")
