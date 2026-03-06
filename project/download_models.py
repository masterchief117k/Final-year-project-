import urllib.request
import os

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

urls = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "openface_nn4.small2.v1.t7": "https://raw.githubusercontent.com/pyimagesearch/face_recognition/master/openface_nn4.small2.v1.t7"
}

for filename, url in urls.items():
    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print("Done.")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

print("Basic Face Detection models checked.")
