import re
import cv2
import numpy as np
from glob import glob

import torch
torch.manual_seed(0)

from sklearn.preprocessing import LabelEncoder
from sklearn import manifold

import matplotlib.pyplot as plt

from vae import get_vae
from simple_vae import get_simple_vae
from visualize_vae import plot_vae_result, img1_convert_to_img2

#Average precision at K
def my_ap_k(point_class, k_nearest_class):
    true_position = np.where(k_nearest_class == point_class)[0]
    ap = 0
    if len(true_position) > 0:
        for i in range(len(true_position)):
            ap += (i+1)/(true_position[i]+1)
        ap /= len(true_position)
    return ap

def calc_map_k(X, y, k = 5):
    ap_array = []
    for i in range(len(X)):
        cur_point = X[i]
        cur_point_class = y[i]
        k_nearest_class = y[np.argsort(np.apply_along_axis(lambda x: np.linalg.norm(cur_point - x), 1, X))[1:k+1]]

        ap_array.append(my_ap_k(cur_point_class, k_nearest_class))
    return np.mean(ap_array)

image_names = np.array(glob("data/*.jpg"))

# название
image_labels = np.array(list(map(lambda image_name: re.sub("data/(.*)_\d\d.jpg", "\g<1>", image_name).lower(), image_names)), dtype = "<U62")

indexes = np.argsort(image_labels)
image_labels = image_labels[indexes]
image_names = image_names[indexes]
unique_labels = np.unique(image_labels)

# ориентация в пространстве 
another_image_labels = np.array(list(map(lambda image_name: re.sub(".*(\d\d)\.jpg", "\g<1>", image_name).lower(), image_names)), dtype = np.int)

# считываем изображения
height, width = 128 * 5, 128 * 9
down_scale = 4
h, w = height // down_scale, width // down_scale
images = np.zeros((len(image_labels), h, w, 3), dtype = np.uint8)
for i, image_name in enumerate(image_names):
    cur_image = plt.imread(image_name)[:, 9:-9]
    cur_height, cur_width = cur_image.shape[:2]
    if cur_height < height:
        cur_image = np.vstack((255 * np.ones((height - cur_height, width, 3), dtype = np.uint8), cur_image))
    else:
        cur_image = cur_image[(cur_height - height)//2:-(cur_height - height)//2, :]
    
    images[i] = cv2.resize(cur_image, (0, 0), fx = 1 / down_scale, fy = 1 / down_scale)


#значения пикселей изображения от -1 до +1, так как потом будем использовать tanh на последнем слое в декодере
X = (images.astype(np.float32) - 128) / 255
X = torch.Tensor(X.reshape(-1, 3, h, w))

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(image_labels)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

simple_vae_128 = get_simple_vae(X, y, 128, epochs=200, device)

plt.subplot(311)
plot_vae_result(X[0], simple_vae_128, device)
plt.subplot(312)
plot_vae_result(X[100], simple_vae_128, device)
plt.subplot(313)
plot_vae_result(X[1000], simple_vae_128, device)
plt.show()

plt.figure(figsize=(18, 4))
plt.subplot(211)
img1_convert_to_img2(X[another_image_labels == 1][1], X[another_image_labels == 1][2], simple_vae_128, device)
plt.subplot(212)
img1_convert_to_img2(X[another_image_labels == 1][1], X[another_image_labels == 10][1], simple_vae_128, device)
plt.show()

res_simple_vae = simple_vae_128
res_latent_size = 128

with torch.no_grad():
    X_simple_encoded = res_simple_vae.encode(X.to(device)).to('cpu').detach().numpy()
if res_latent_size > 2:
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_simple_tsne = tsne.fit_transform(X_simple_encoded)
else:
    X_simple_tsne = np.copy(X_simple_encoded)

plt.figure(figsize = (30, 15))
plt.subplot(121)
# кластера классов
plt.scatter(X_simple_tsne[:, 0], X_simple_tsne[:, 1], c = plt.cm.get_cmap('hsv', len(np.unique(y)))(y))
plt.subplot(122)
# кластера ориентаций
plt.scatter(X_simple_tsne[:, 0], X_simple_tsne[:, 1], c = plt.cm.get_cmap('hsv', len(np.unique(another_image_labels)))(another_image_labels))
plt.show()


# В итоге хорошо различаем модели!
k = 5
print(f"Map {k} : {calc_map_k(X_simple_encoded, y, k)}")
# ориентацию не очень хорошо различаем (может быть потому что надо сделать их меньше)
k = 5
print(f"Map {k} : {calc_map_k(X_simple_encoded, another_image_labels, k)}")

t = np.unique(another_image_labels)
t = np.append(t[2:], t[:2])

new_another_image_labels = np.zeros_like(another_image_labels)
for i, l in enumerate(np.split(t, 9)):
    new_another_image_labels[(another_image_labels.reshape(-1, 1) == l).any(1)] = i

# другое дело
k = 5
print(f"Map {k} : {calc_map_k(X_simple_encoded, new_another_image_labels, k)}")

h, w = height // down_scale, width // down_scale
min_point = np.min(X_simple_tsne, axis = 0)
max_point = np.max(X_simple_tsne, axis = 0)

number_of_ceils = 25
x_range = np.linspace(min_point[0], max_point[0], number_of_ceils)
y_range = np.linspace(min_point[1], max_point[1], number_of_ceils)
image_map = np.zeros(((number_of_ceils - 1) * h, (number_of_ceils - 1) * w, 3), dtype = np.uint8)
for y in range(number_of_ceils-1):
    for x in range(number_of_ceils-1):
        left_down_corner = np.array([x_range[x], y_range[y]])
        right_up_corner = np.array([x_range[x+1], y_range[y+1]])

        mask = np.logical_and((X_simple_tsne >= left_down_corner).all(1), (X_simple_tsne < right_up_corner).all(1))
        if mask.sum() > 0:
            image_map[(number_of_ceils - 1) * h - (y+1)*h :(number_of_ceils - 1) * h - y * h,  x * w : (x + 1) * w] = images[mask][np.random.choice(np.arange(len(images[mask])))]

plt.figure(figsize=(30, 30))
plt.imshow(image_map)
plt.show()
