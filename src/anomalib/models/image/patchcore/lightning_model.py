import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize, Grayscale
from collections.abc import Sequence
from typing import Any

import logging
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import PatchcoreModel

logger = logging.getLogger(__name__)


class Patchcore(MemoryBankMixin, AnomalibModule):
    """
    Lightning module implementing PatchCore anomaly detection.

    - Extracts embeddings from a pretrained YOLO-backed PatchcoreModel.
    - Builds a memory bank via coreset sampling.
    - Performs one-class inference to detect anomalies.
    - Optionally exports the traced model as TorchScript on training end.
    """

    def __init__(
        self,
        layers: Sequence[str] = ("seg_mask",),
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        pre_processor: nn.Module | bool = False,  # Disable PreProcessor
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
        export_pt_path: str | None = None,
        model_path: str | None = None,
    ) -> None:
        """
        Initialize the Patchcore Lightning module.

        Args:
            layers: Which feature map names to extract from the backbone.
            coreset_sampling_ratio: Fraction of embeddings to keep during coreset sampling.
            num_neighbors: Number of nearest neighbors for anomaly scoring.
            pre_processor: PreProcessor instance or False to disable.
            post_processor: PostProcessor instance or True to use default.
            evaluator: Evaluator instance or True to use default.
            visualizer: Visualizer instance or True to use default.
            export_pt_path: If provided, path to save TorchScript model after training.
            model_path: Path to the pretrained YOLO-backed model TorchScript file.
        """
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.model = PatchcoreModel(
            layers=layers,
            model_path=model_path,
            num_neighbors=num_neighbors,
            target_size=(224, 224),  # Match YOLO input size
        )
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.embeddings: list[torch.Tensor] = []
        self.export_pt_path = export_pt_path

    @staticmethod
    def configure_optimizers() -> None:
        """
        Disable optimizers—PatchCore uses no gradient-based training.
        """
        return None

    def training_step(self, batch: Batch, *args, **kwargs) -> None:
        """
        Extract embeddings on each training batch and store them.

        Args:
            batch: A Batch object containing input images.
        Returns:
            Dummy zero loss tensor on GPU to satisfy Lightning’s API.
        """
        del args, kwargs
        embedding = self.model(batch.image)
        # Keep embedding on GPU, append directly
        self.embeddings.append(embedding)
        print("appending embeddings")
        return torch.tensor(0.0, requires_grad=True, device="cuda")  # Return on GPU

    def fit(self) -> None:
        """
        After collecting all embeddings, perform coreset sampling to build the memory bank.
        """
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings).cuda()  # Stack and move to GPU
        logger.info("Applying core-set subsampling to get the embedding.")
        self.model.subsample_embedding(embeddings, self.coreset_sampling_ratio)

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """
        Run inference on validation images to produce anomaly scores and maps.

        Args:
            batch: A Batch object containing validation images.
        Returns:
            Updated Batch with prediction fields (scores, maps).
        """
        del args, kwargs
        predictions = self.model(batch.image)
        return batch.update(**predictions)

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """
        Provide custom Lightning trainer arguments.

        Returns:
            Dict overriding default trainer settings.
        """
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """
        Specify that this is a one-class learning problem.
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        """
        Instantiate the default PostProcessor for anomaly map smoothing and thresholding.
        """
        return PostProcessor()

    def export_torchscript(
        self,
        output_path: str,
        input_shape: tuple[int, int, int, int] = (1, 1, 224, 224),
    ) -> None:
        """
        Trace and save the PatchcoreModel as a TorchScript .pt file.

        Args:
            output_path: File path to write the traced model.
            input_shape: Shape of dummy tensor for tracing (N, C, H, W).
        """
        # Ensure the model is in eval mode and on the appropriate device
        self.model.eval().to(self.device)

        # Dummy input matching the preprocessing
        dummy = torch.randn(input_shape, device=self.device)

        # Trace and save
        traced = torch.jit.trace(self.model, dummy, strict=False)
        traced.save(output_path)
        print(f"Saved TorchScript model to {output_path}")

    def on_train_end(self) -> None:
        """
        Lightning hook at end of training—automatically export if path provided.
        """
        # Automatically export .pt if path provided
        if self.export_pt_path:
            self.export_torchscript(self.export_pt_path)
