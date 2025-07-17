from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Resize

from anomalib.models.components import DynamicBufferMixin, KCenterGreedy
from .anomaly_map import AnomalyMapGenerator
from annoy import AnnoyIndex

if TYPE_CHECKING:
    from anomalib.data.utils.tiler import Tiler


class PatchcoreModel(DynamicBufferMixin, nn.Module):
    """
    Core PatchCore model that:
      - Loads a pretrained TorchScript feature extractor.
      - Extracts patch embeddings from input images.
      - Builds a memory bank via coreset sampling.
      - Computes anomaly maps and scores at inference time.
    """

    def __init__(
        self,
        layers: Sequence[str] = ["seg_mask"],
        model_path: str | None = None,
        num_neighbors: int = 9,
        target_size: tuple[int, int] = (224, 224),
    ) -> None:
        """
        Initialize the PatchcoreModel.

        Args:
            layers: Names of feature maps to use for embedding.
            model_path: Path to the pretrained TorchScript model.
            num_neighbors: Number of neighbors to query for scoring.
            target_size: Spatial size to which inputs are resized.
        """
        super().__init__()
        self.tiler: Tiler | None = None
        self.layers = layers
        self.num_neighbors = num_neighbors
        self.target_size = target_size

        # Load and move TorchScript model to GPU directly
        self.feature_extractor = torch.jit.load(model_path, map_location="cuda").eval()
        self.annoy_index = None
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator()
        self.resize_transform = Resize(target_size, antialias=True)

        self.register_buffer("memory_bank", torch.Tensor())
        self.memory_bank: torch.Tensor

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Forward pass through the PatchcoreModel.

        - Applies tiling (if configured).
        - Normalizes and resizes input.
        - Runs the feature extractor to get patch features.
        - Pools features and generates embeddings.
        - During inference, computes nearest-neighbor distances,
          constructs an anomaly map, and returns scores.

        Args:
            input_tensor: Batch of input images, shape (N, C, H, W).

        Returns:
            If training: raw embeddings tensor.
            If eval: dict with "anomaly_map" and "pred_score".
        """
        output_size = input_tensor.shape[-2:]
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        # Preprocess input on GPU
        input_tensor = input_tensor.cuda()  # Explicitly move to GPU
        input_tensor = (input_tensor - 0.5) / 0.5  # Normalize
        input_tensor = self.resize_transform(input_tensor)  # Resize to target_size

        # Ensure input has 1 channel as expected by the model
        if input_tensor.shape[1] != 1:
            input_tensor = input_tensor.mean(dim=1, keepdim=True)  # Convert 3-channel to 1-channel by averaging

        with torch.no_grad():
            try:
                classification_result, seg_mask_result = self.feature_extractor(input_tensor)
                features = {self.layers[0]: seg_mask_result}
            except Exception as e:
                raise RuntimeError(f"Error in feature_extractor: {e}")

        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        if self.training:
            output = embedding
        else:
            patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
            patch_scores = patch_scores.reshape((batch_size, -1))
            locations = locations.reshape((batch_size, -1))

            pred_score = self.compute_anomaly_score(patch_scores, locations, embedding)
            patch_scores = patch_scores.reshape((batch_size, 1, width, height))
            anomaly_map = self.anomaly_map_generator(patch_scores, output_size)
            output = {"anomaly_map": anomaly_map, "pred_score": pred_score}

        return output

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract the embedding tensor from pooled features.

        Args:
            features: Dict mapping layer names to tensors of shape (N, C, H, W).

        Returns:
            Embedding tensor of shape (N, C, H, W).
        """
        embeddings = features[self.layers[0]]
        return embeddings

    @staticmethod
    def reshape_embedding(embedding: torch.Tensor) -> torch.Tensor:
        """
        Flatten spatial dimensions into a patch sequence.

        Args:
            embedding: Tensor of shape (N, C, H, W).

        Returns:
            Tensor of shape (N*H*W, C).
        """
        N, C, H, W = embedding.shape
        return embedding.permute(0, 2, 3, 1).reshape(-1, C)

    def subsample_embedding(self, embedding: torch.Tensor, sampling_ratio: float) -> None:
        """
        Build the memory bank using coreset sampling and Annoy.

        Args:
            embedding: Full set of extracted embeddings.
            sampling_ratio: Fraction to keep for the coreset.
        """
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
        coreset = sampler.sample_coreset()
        self.memory_bank = coreset.cuda()  # Move memory_bank to GPU

        embedding_size = self.memory_bank.shape[1]
        self.annoy_index = AnnoyIndex(embedding_size, metric="euclidean")
        for i, vec in enumerate(self.memory_bank):
            self.annoy_index.add_item(i, vec.cpu().numpy())
        print("annoy tree built")
        self.annoy_index.build(10)

    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Euclidean distances between x and y.

        Args:
            x: Tensor of shape (M, D).
            y: Tensor of shape (N, D).

        Returns:
            Distances tensor of shape (M, N).
        """
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)
        dist2 = x_norm - 2 * x @ y.transpose(-2, -1) + y_norm.transpose(-2, -1)
        return dist2.clamp_min_(0).sqrt_()

    def nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int) -> tuple:
        """
        Query the Annoy index for nearest neighbors.

        Args:
            embedding: Tensor of shape (M, D) on CPU or GPU.
            n_neighbors: Number of nearest neighbors to retrieve.

        Returns:
            Tuple (distances, indices), each of shape (M, n_neighbors).
        """
        distances = []
        indices = []
        for vector in embedding.cpu().numpy():  # Move to CPU for Annoy
            nearest_indices, nearest_distances = self.annoy_index.get_nns_by_vector(
                vector, n_neighbors, include_distances=True
            )
            distances.append(nearest_distances)
            indices.append(nearest_indices)
        distances = torch.tensor(distances, device=embedding.device)
        indices = torch.tensor(indices, device=embedding.device)
        return distances, indices

    def compute_anomaly_score(
        self,
        patch_scores: torch.Tensor,
        locations: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a final anomaly score per image, possibly using multiple neighbors.

        - For num_neighbors=1, returns the max patch score.
        - Otherwise, weights the top patch score by support neighbor distances.

        Args:
            patch_scores: Tensor (batch_size, num_patches).
            locations: Tensor of nearest-neighbor indices.
            embedding: Flattened embedding tensor (batch_size*num_patches, D).

        Returns:
            Tensor of shape (batch_size,) with final anomaly scores.
        """
        batch_size, num_patches = patch_scores.shape

        if self.num_neighbors == 1:
            return patch_scores.amax(dim=1)

        # Identify most anomalous patch per sample
        max_patches = torch.argmax(patch_scores, dim=1)
        score = patch_scores[torch.arange(batch_size), max_patches]
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]

        # Gather neighbor samples from memory bank
        nn_index = locations[torch.arange(batch_size), max_patches].cpu()
        nn_sample = self.memory_bank[nn_index].to(embedding.device)

        # Compute additional neighbor distances and weights
        _, support_samples = self.nearest_neighbors(
            embedding=nn_sample, n_neighbors=min(self.num_neighbors, len(self.memory_bank))
        )
        distances = self.euclidean_dist(
            max_patches_features.unsqueeze(1), self.memory_bank[support_samples.cpu()].to(embedding.device)
        )
        weights = (1 - F.softmax(distances.squeeze(1), dim=1))[:, 0]
        return weights * score
