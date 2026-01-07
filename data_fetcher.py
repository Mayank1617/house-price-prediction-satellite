import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import tqdm

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
IMAGE_SIZE = '400x400'
ZOOM = 19

BASE_DIR = "../data/raw"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "images_train")
TEST_IMG_DIR = os.path.join(BASE_DIR, "images_test")

df_train = pd.read_excel(os.path.join(BASE_DIR,"train1.xlsx"))
df_test = pd.read_excel(os.path.join(BASE_DIR,"test2.xlsx"))

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(TEST_IMG_DIR, exist_ok=True)

# FUNCTION FOR IMAGE FETCHING
def fetch_satellite_image(lat, long, save_path):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{long}&zoom={ZOOM}&size={IMAGE_SIZE}&maptype=satellite&key={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(save_path)
            return True
        else:
            print(f"Failed for {lat}, {long}: {response.status_code}")
    except Exception as e:
        print(f"Error fetching {lat}, {long}: {e}")
    return False


# TRAINING IMAGES FETCHING
success_count = 0
fail_count = 0
for idx, row in tqdm.tqdm(df_train.iterrows(), desc="Downloading train images"):
    img_path = os.path.join(TRAIN_IMG_DIR, f"{row['id']}.png")
    if not os.path.exists(img_path):
        if fetch_satellite_image(row['lat'], row['long'], img_path):
            success_count += 1
        else:
            fail_count += 1
print(f"Train images: {success_count} downloaded, {fail_count} failed")


# TEST IMAGES FETCHING
success_count = 0
fail_count = 0
for idx, row in tqdm.tqdm(df_test.iterrows(), desc="Downloading test images"):
    img_path = os.path.join(TEST_IMG_DIR, f"{row['id']}.png")
    if not os.path.exists(img_path):
        if fetch_satellite_image(row['lat'], row['long'], img_path):
            success_count += 1
        else:
            fail_count += 1
print(f"Test images: {success_count} downloaded, {fail_count} failed")