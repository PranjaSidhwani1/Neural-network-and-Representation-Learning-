import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import norm
from skimage.metrics import structural_similarity as ssim

# from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# class VAE(nn.Module):
#     def __init__(self, input_dim=784, latent_dim=2):
#         super(VAE, self).__init__()
#         # self.encoder = nn.Sequential(
#         #     nn.Linear(input_dim, 400),
#         #     nn.ReLU(),
#         #     nn.Linear(400, latent_dim * 2),  # mean and log-variance
#         # )
#         # self.decoder = nn.Sequential(
#         #     nn.Linear(latent_dim, 400),
#         #     nn.ReLU(),
#         #     nn.Linear(400, input_dim),
#         #     nn.Sigmoid(),
#         # )

#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.BatchNorm1d(512),
#             nn.Linear(512, 400),
#             nn.ReLU(),
#             nn.BatchNorm1d(400),
#             nn.Linear(400, latent_dim * 2),  # mean and log-variance
#         )

#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 400),
#             nn.ReLU(),
#             nn.BatchNorm1d(400),
#             nn.Linear(400, 512),
#             nn.ReLU(),
#             nn.BatchNorm1d(512),
#             nn.Linear(512, input_dim),
#             nn.Sigmoid(),
#         )

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         h = self.encoder(x)
#         mu, logvar = torch.chunk(h, 2, dim=-1)
#         z = self.reparameterize(mu, logvar)
#         return self.decoder(z), mu, logvar


class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=2, kl_anneal_steps=1000):
        super(VAE, self).__init__()

        self.kl_anneal_steps = kl_anneal_steps  # Steps over which to anneal KL term
        self.global_step = 0  # Track the training step for KL annealing

        # Encoder with BatchNorm and Dropout layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 400),
            nn.ReLU(),
            nn.BatchNorm1d(400),
            nn.Dropout(0.2),
            nn.Linear(400, latent_dim * 2),  # mean and log-variance
        )

        # Decoder with BatchNorm and Dropout
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.BatchNorm1d(400),
            nn.Dropout(0.2),
            nn.Linear(400, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

        # Apply Xavier initialization for better training stability
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)  # Split mean and logvar
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # Binary cross-entropy for reconstruction loss
        recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")

        # KL Divergence with annealing
        kl_weight = min(1.0, self.global_step / self.kl_anneal_steps)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.global_step += 1  # Increment training step for annealing

        return recon_loss + kl_weight * kl_divergence


class CustomGMM:
    def __init__(self, n_components, n_iter=100, tol=1e-3):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.weights = None
        self.means = None
        self.covariances = None
        self.cluster_mapping = None  # To map clusters to actual class labels

    def fit(self, X, train_loader):
        # Extract labels from train_loader (use it for dynamic class mapping)
        labels = []
        for _, label in train_loader:
            labels.append(label.numpy())
        labels = np.concatenate(labels, axis=0)

        # Get unique labels in the training dataset to create a dynamic mapping
        unique_labels = np.unique(labels)
        self.cluster_mapping = {i: label for i, label in enumerate(unique_labels)}
        # print("Cluster Mapping:", self.cluster_mapping)

        n_samples, n_features = X.shape
        self.weights = np.full(self.n_components, 1 / self.n_components)
        rng = np.random.default_rng(seed=0)
        self.means = X[rng.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.eye(n_features)] * self.n_components)

        log_likelihood_old = None

        for iteration in range(self.n_iter):
            # E-step: Calculate responsibilities
            responsibilities = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                resp_k = self.weights[k] * self._gaussian_pdf(
                    X, self.means[k], self.covariances[k]
                )
                responsibilities[:, k] = resp_k
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)

            # M-step: Update weights, means, and covariances
            N_k = responsibilities.sum(axis=0)
            self.weights = N_k / n_samples
            self.means = (responsibilities.T @ X) / N_k[:, None]
            for k in range(self.n_components):
                X_centered = X - self.means[k]
                self.covariances[k] = (
                    (responsibilities[:, k, None] * X_centered).T @ X_centered / N_k[k]
                )
                self.covariances[k].flat[
                    :: n_features + 1
                ] += self.tol  # Regularization

            # Check convergence
            log_likelihood = np.sum(np.log(np.sum(responsibilities, axis=1)))
            if (
                log_likelihood_old is not None
                and abs(log_likelihood - log_likelihood_old) < self.tol
            ):
                break
            log_likelihood_old = log_likelihood

    def _gaussian_pdf(self, X, mean, cov):
        n_features = X.shape[1]
        cov_inv = np.linalg.inv(cov)
        diff = X - mean
        exp_term = np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))
        return exp_term / np.sqrt((2 * np.pi) ** n_features * np.linalg.det(cov))

    def predict(self, X):
        likelihoods = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            likelihoods[:, k] = self.weights[k] * self._gaussian_pdf(
                X, self.means[k], self.covariances[k]
            )
        predicted_clusters = np.argmax(likelihoods, axis=1)
        # Map the predicted clusters to the learned labels from the training dataset
        return np.array(
            [self.cluster_mapping[cluster] for cluster in predicted_clusters]
        )


def get_data_loader(data_path, batch_size=64, shuffle=True):
    data = np.load(data_path)
    images, labels = data["data"], data["labels"]

    images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(images, labels)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


# Loss Function for VAE
def vae_loss(recon_x, x, mu, logvar):
    # recon_x = torch.clamp(recon_x, min=1e-5, max=1 - 1e-5)
    recon_loss = nn.functional.mse_loss(recon_x, x.view(-1, 784), reduction="sum")
    # recon_loss = nn.functional.binary_cross_entropy_with_logits(
    #     recon_x, x.view(-1, 784)
    # )
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # * 10
    return recon_loss + kl_div


def plot_2d_manifold(vae, latent_dim=2, n=20, digit_size=28, device="cuda"):
    figure = np.zeros((digit_size * n, digit_size * n))

    # Generate a grid of values between 0.05 and 0.95 percentiles of a normal distribution
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    vae.eval()  # Set VAE to evaluation mode
    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]], device=device).float()

                digit = (
                    vae.decoder(z_sample).cpu().numpy().reshape(digit_size, digit_size)
                )

                figure[
                    i * digit_size : (i + 1) * digit_size,
                    j * digit_size : (j + 1) * digit_size,
                ] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="gnuplot2")
    plt.axis("off")
    plt.show()
    plt.savefig("2d_manifold.png")


def visualize_latent_space(model, train_loader):
    latent_vectors = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch, label in train_loader:
            batch = batch.to(device)
            _, mu, _ = model(batch)
            latent_vectors.append(mu.cpu().numpy())
            labels.append(label.numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        latent_vectors[:, 0], latent_vectors[:, 1], c=labels, cmap="viridis", alpha=0.5
    )
    plt.colorbar(scatter, label="Class Label")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Latent Space Visualization")
    plt.savefig("latent_space.png")
    plt.show()


# Train VAE and GMM
def train(train_loader, val_loader, model, epochs, save_path_vae, save_path_gmm):
    # Training VAE using train_loader
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # weight_decay=1e-5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch, _ in train_loader:
            batch = batch.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, logvar = model(batch)

            # Compute the loss
            # loss = vae_loss(recon_batch, batch, mu, logvar)
            loss = model.loss_function(recon_batch, batch, mu, logvar)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader.dataset)}"
        )

    # Save VAE model
    torch.save(model.state_dict(), save_path_vae)

    # Collect latent space embeddings for GMM
    latent_vectors = []
    model.eval()
    with torch.no_grad():
        for batch, _ in val_loader:
            batch = batch.to(device)
            _, mu, _ = model(batch)
            latent_vectors.append(mu.cpu().numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)

    # Now, train the GMM using train_loader
    gmm = CustomGMM(n_components=3)
    gmm.fit(latent_vectors, train_loader)

    # Save the GMM model
    with open(save_path_gmm, "wb") as f:
        pickle.dump(gmm, f)

    # For Report
    show_reconstruction(vae, val_loader)
    plot_2d_manifold(vae)
    visualize_latent_space(vae, train_loader)


def show_reconstruction(model, val_loader, n=15):
    model.eval()
    data, labels = next(iter(val_loader))

    data = data.to(device)
    recon_data, _, _ = model(data)

    fig, axes = plt.subplots(2, n, figsize=(15, 4))
    for i in range(n):
        # Original images
        axes[0, i].imshow(data[i].cpu().numpy().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        # Reconstructed images
        axes[1, i].imshow(
            recon_data[i].cpu().view(28, 28).detach().numpy(), cmap="gray"
        )
        axes[1, i].axis("off")
    plt.show()
    plt.savefig("reconstruction.png")


# Reconstruct images using VAE
def reconstruct(test_loader, model, save_path):
    all_images = []
    for batch, _ in test_loader:
        all_images.append(batch.cpu().numpy())
    all_images = np.concatenate(all_images, axis=0).reshape(-1, 28, 28)

    # # DEBUG: Display all test samples in one image
    # fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    # for i, ax in enumerate(axes.flat):
    #     if i < len(all_images):
    #         ax.imshow(all_images[i], cmap="gray")
    #     ax.axis("off")
    # plt.savefig("all_test_samples.png")

    model.eval()
    reconstructions = []
    with torch.no_grad():
        for batch, _ in test_loader:
            batch = batch.to(device)
            recon_batch, _, _ = model(batch)
            reconstructions.append(recon_batch.cpu().numpy())

    # # DEBUG: Display all reconstructed images
    reconstructions = np.concatenate(reconstructions, axis=0).reshape(-1, 28, 28)
    # fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    # for i, ax in enumerate(axes.flat):
    #     if i < len(reconstructions):
    #         ax.imshow(reconstructions[i], cmap="gray")
    #     ax.axis("off")
    # plt.savefig("all_reconstructions.png")

    # DEBUG: Calculate reconstruction error for each sample
    reconstruction_errors = []
    for original, reconstructed in zip(all_images, reconstructions):
        error = ssim(
            original, reconstructed, data_range=original.max() - original.min()
        )
        reconstruction_errors.append(error)

    print("Reconstruction Errors:", reconstruction_errors)
    print("Mean Reconstruction Error:", np.mean(reconstruction_errors))

    reconstructions = np.concatenate(reconstructions, axis=0)
    np.save(save_path, reconstructions)


# Classify test samples using VAE and GMM
def classify(test_loader, model, gmm_path):
    model.eval()
    with open(gmm_path, "rb") as f:
        gmm = pickle.load(f)

    latent_vectors = []
    with torch.no_grad():
        for batch, _ in test_loader:
            batch = batch.to(device)
            _, mu, _ = model(batch)
            latent_vectors.append(mu.cpu().numpy())
    latent_vectors = np.concatenate(latent_vectors, axis=0)

    predictions = gmm.predict(latent_vectors)
    print("Predictions:", predictions)

    # DEBUG: Calculate prediction accuracy
    actual = [1] * 5 + [4] * 5 + [8] * 5
    print("Prediction Accuracy:", np.mean(predictions == actual))

    # DEBUG: Calculate prediction confidence
    confidence = []
    for i, prediction in enumerate(predictions):
        prediction_to_class = {v: k for k, v in gmm.cluster_mapping.items()}
        # print("Inverted Cluster Mapping:", prediction_to_class)
        class_label = prediction_to_class[prediction]
        likelihoods = gmm._gaussian_pdf(
            latent_vectors, gmm.means[class_label], gmm.covariances[class_label]
        )
        # print("Likelihoods:", likelihoods)
        confidence.append(likelihoods[prediction])
    # for i, latent_vector in enumerate(latent_vectors):
    #     likelihoods = [
    #         gmm.weights[k]
    #         * gmm._gaussian_pdf(
    #             latent_vector.reshape(1, -1), gmm.means[k], gmm.covariances[k]
    #         )[0]
    #         for k in range(gmm.n_components)
    #     ]
    #     # Normalize the likelihood of the predicted class
    #     confidence_value = likelihoods[predictions[i]] / sum(likelihoods)
    #     confidence.append(confidence_value)

    print("Prediction Confidence:", confidence)
    print("Mean Prediction Confidence:", np.mean(confidence))


if __name__ == "__main__":
    # Reproducibility
    seed_value = 0
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Parse command-line arguments
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3] if len(sys.argv) > 3 else None
    arg4 = sys.argv[4] if len(sys.argv) > 4 else None
    arg5 = sys.argv[5] if len(sys.argv) > 5 else None

    if len(sys.argv) == 4:  # Running code for VAE reconstruction.
        path_to_test_dataset = arg1
        command = arg2
        vaePath = arg3

    elif len(sys.argv) == 5:  # Running code for class prediction during testing
        path_to_test_dataset = arg1
        command = arg2
        vaePath = arg3
        gmmPath = arg4

    else:  # Running code for training
        path_to_train_dataset = arg1
        path_to_val_dataset = arg2
        command = arg3
        vaePath = arg4
        gmmPath = arg5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    if command == "train":
        train_loader = get_data_loader(path_to_train_dataset)
        val_loader = get_data_loader(path_to_val_dataset, shuffle=False)
    elif command in ["test_reconstruction", "test_classifier"]:
        test_loader = get_data_loader(path_to_test_dataset, batch_size=1, shuffle=False)

    # Initialize VAE model
    vae = VAE().to(device)
    num_params = sum(p.numel() for p in vae.parameters())
    print(f"Total number of parameters: {num_params}; Valid? {num_params < 2e7}")

    if command == "train":
        train(
            train_loader,
            val_loader,
            vae,
            epochs=100,
            save_path_vae=vaePath,
            save_path_gmm=gmmPath,
        )
    elif command == "test_reconstruction":
        vae.load_state_dict(torch.load(vaePath, map_location=device))
        reconstruct(test_loader, vae, save_path="reconstructions.npy")
    elif command == "test_classifier":
        vae.load_state_dict(torch.load(vaePath, map_location=device))
        classify(test_loader, vae, gmmPath)
