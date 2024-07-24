import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from matplotlib.offsetbox import (
    OffsetImage, AnnotationBbox)
from otpca import ot_pca_bcd

dataset = datasets.MNIST(
    root='/home/efernand/data/', download=True, train=True)

ims = dataset.data
X = (ims.to(torch.float32) / 255).reshape(-1, 28 ** 2)
labels = dataset.targets

ind = np.random.choice(np.arange(len(X)), size=1000)
X = X[ind]
labels = labels[ind]

Gbcd, Pbcd, log_bcd = ot_pca_bcd(
    X.numpy(), k=2, reg=1, verbose=True,
    method='MM', svd_fct_cpu='numpy',
    max_iter_sink=100)
Y = np.dot(X.numpy(), Pbcd)

plt.figure(figsize=(5, 5))
for yu in np.unique(labels):
    ind = np.where(labels == yu)[0]
    plt.scatter(Y[ind, 0], Y[ind, 1], label=f"Digit {yu}", cmap='tab10')
plt.legend(bbox_to_anchor=(1, 1))

i = 0
plt.scatter(Y[i, 0], Y[i, 1], c='k', s=50, marker='*')
plt.show()

# Create a scatter plot
fig, ax = plt.subplots(figsize=(10, 10))

# Add images to scatter plot
for i in range(0, len(X)):
    # Create an OffsetImage object
    imagebox = OffsetImage(X[i].reshape(28, 28), zoom=0.5, cmap='gray')

    # Create an AnnotationBbox object
    ab = AnnotationBbox(imagebox, (Y[i, 0], Y[i, 1]), frameon=False)

    # Add the AnnotationBbox to the axes
    ax.add_artist(ab)

# Set limits and show plot
ax.set_xlim(Y[:, 0].min() - 1, Y[:, 0].max() + 1)
ax.set_ylim(Y[:, 1].min() - 1, Y[:, 1].max() + 1)
plt.show()
