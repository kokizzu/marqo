from abc import ABC, abstractmethod
from typing import Optional

from marqo.tensor_search.models.private_models import ModelAuth


class AbstractEmbeddingModel(ABC):
    """This is the abstract base class for all models in Marqo."""

    def __init__(self, model_properties: Optional[dict] = None, device: Optional[str] = None,
                 model_auth: Optional[ModelAuth] = None):
        """Load the model with the given properties.

        Args:
            model_properties (dict): The properties of the model.
            device (str): The device to load the model on.
            model_auth (dict): The authentication information for the model.
        """
        if device is None:
            raise ValueError("`device` is required for loading CLIP models!")

        if model_properties is None:
            model_properties = dict()

        self.device = device
        self.model_auth = model_auth

    def load(self):
        """Load the model and check if the necessary component are loaded.

        The required components are loaded in the `_load_necessary_components` method.
        The loaded components are checked in the `_check_loaded_components` method.
        """
        self._load_necessary_components()
        self._check_loaded_components()

    @abstractmethod
    def _load_necessary_components(self):
        """Load the necessary components for the model."""
        pass

    @abstractmethod
    def _check_loaded_components(self):
        """Check if the necessary components are loaded.

        Raises:
            A proper exception if the necessary components are not loaded.
        """
        pass

    @abstractmethod
    def encode(self):
        """Encode the input data."""
        pass