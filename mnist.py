
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from matplotlib.offsetbox import (
    OffsetImage, AnnotationBbox)

from gw_mds import GromovWassersteinMultiDimensionalScaling

dataset = datasets.MNIST(
    root='/home/efernand/data/', download=True, train=True)

ims = dataset.data
X = (ims.to(torch.float32) / 255).reshape(-1, 28 ** 2)
labels = dataset.targets

ind = np.random.choice(np.arange(len(X)), size=1000)
X = X[ind]
labels = labels[ind]

gw_mds = GromovWassersteinMultiDimensionalScaling(
    n_components=2,
    init='pca',
    optimizer_name='adam',
    learning_rate=0.1,
    metric_fn=None,
    precomputed_metric=False
)
gw_mds.fit(X, n_iter=100)
Y = gw_mds.embeddings_.clone().detach()

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
