import torch
import matplotlib.pyplot as plt

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
img = 'https://i.ytimg.com/vi/q71MCWAEfL8/maxresdefault.jpg'  # or file, Path, PIL, OpenCV, numpy, list
results = model(img)
results.save(save_dir='results')
fig, ax = plt.subplots(figsize=(16, 12))
ax.imshow(results.render()[0])
plt.show()