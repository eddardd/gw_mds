import ot
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA


class GromovWassersteinMultiDimensionalScaling(torch.nn.Module):
    def __init__(self,
                 n_components=2,
                 init='randn',
                 optimizer_name='adam',
                 learning_rate=0.1,
                 metric_fn=None,
                 precomputed_metric=False):
        super(GromovWassersteinMultiDimensionalScaling, self).__init__()

        assert init.lower() in ['randn', 'rand', 'pca'], (
            "Expected 'init' to be either 'randn', 'rand' or"
            f" 'pca', but got {init.lower()}"
        )

        assert optimizer_name.lower() in ['sgd', 'adam'], (
            "Expected 'optimizer_name' to be either 'sgd' or"
            f" 'adam', but got {optimizer_name.lower()}"
        )

        self.n_components = n_components
        self.init = init.lower()
        self.learning_rate = learning_rate
        self.embeddings_ = None
        self.fitted = False
        self.optimizer_name = optimizer_name.lower()
        self.precomputed_metric = precomputed_metric

        if metric_fn is None:
            self.metric_fn = torch.cdist
        else:
            self.metric_fn = metric_fn

        self.history = {
            'loss': [],
            'embeddings': []
        }

    def compute_gw_loss(self, DX, DY):
        u = torch.ones(len(DX)) / len(DX)
        with torch.no_grad():
            T = ot.gromov.gromov_wasserstein(
                DX, DY, u, u, loss_fun='square_loss',
                verbose=False, tol=1e-5)
        constC, hX, hY = ot.gromov._utils.init_matrix(
            DX, DY, u, u, loss_fun='square_loss')
        return (ot.gromov._utils.gwloss(constC, hX, hY, T), T)

    def fit(self, X, n_iter=100):
        n, _ = X.shape
        if self.init == 'randn':
            self.embeddings_ = torch.randn(n, self.n_components)
        elif self.init == 'rand':
            self.embeddings_ = torch.rand(n, self.n_components)
        elif self.init == 'pca' and not self.precomputed_metric:
            X_pca = PCA(n_components=self.n_components).fit_transform(
                X.numpy())
            X_pca = torch.from_numpy(X_pca).float()
            self.embeddings_ = X_pca
        self.embeddings_.requires_grad = True

        if self.optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(
                [self.embeddings_,], lr=self.learning_rate, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(
                [self.embeddings_,], lr=self.learning_rate)

        if self.precomputed_metric:
            # In this case, X is itself the matrix of
            # pairwise distance
            DX = X.clone()
        else:
            DX = self.metric_fn(X, X)

        # Fit loop
        pbar = tqdm(range(n_iter))
        for it in pbar:
            self.optimizer.zero_grad()
            DY = torch.cdist(
                self.embeddings_, self.embeddings_, p=2)
            loss, T = self.compute_gw_loss(DX, DY)
            loss.backward()
            self.optimizer.step()
            pbar.set_description(
                f"[{it}] Loss: {loss.item()}")

            self.history['loss'].append(loss.item())
            self.history['embeddings'].append(
                self.embeddings_.detach().clone())

        # Re-orders the embeddings
        T = T.detach().clone()
        T /= T.sum(dim=0)[None, :]
        with torch.no_grad():
            self.embeddings_.data = torch.matmul(
                T, self.embeddings_.data)

        self.fitted = True
        return self

    def fit_transform(self, X, n_iter=100):
        self.fit(X, n_iter)
        return self.ebemddings_.detach().clone()
