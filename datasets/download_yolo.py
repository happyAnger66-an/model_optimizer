import kagglehub

# Download latest version
path = kagglehub.dataset_download("ultralytics/coco128")

print("Path to dataset files:", path)
