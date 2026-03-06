import urllib.request
import os

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

urls = {
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
    "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
}

for filename, url in urls.items():
    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Done downloading {filename}.")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

print("YOLO models checked.")
