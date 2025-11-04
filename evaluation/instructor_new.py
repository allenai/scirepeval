from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
import importlib.metadata
import warnings
from string import Formatter

# Lazy import for optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Version requirements
MIN_TRANSFORMERS_VERSION_QWEN3 = "4.51.0"
MIN_TRANSFORMERS_VERSION_GEMMA = "4.56.0"
MIN_SENTENCE_TRANSFORMERS_VERSION = "2.7.0"


def _parse_version(version_str: str) -> tuple:
    """Parse version string into tuple of integers for comparison."""
    try:
        version_str = version_str.split('-')[0]
        return tuple(int(x) for x in version_str.split('.'))
    except (ValueError, AttributeError):
        return (0, 0, 0)


def _get_package_version(package_name: str) -> str:
    """Get installed version of a package."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _check_version_compatibility(model_type: str) -> tuple:
    """
    Check if the installed transformers version supports the requested model type.

    Returns:
        (is_compatible: bool, error_message: str)
    """
    transformers_version = _get_package_version("transformers")
    current_version = _parse_version(transformers_version)

    if model_type == "qwen3":
        required_version = _parse_version(MIN_TRANSFORMERS_VERSION_QWEN3)
        if current_version < required_version:
            return False, (
                f"Qwen3 requires transformers >= {MIN_TRANSFORMERS_VERSION_QWEN3}, "
                f"but you have {transformers_version}. "
                f"Please upgrade: pip install 'transformers>={MIN_TRANSFORMERS_VERSION_QWEN3}'"
            )

    elif model_type == "gemma":
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return False, (
                f"Gemma requires sentence-transformers to be installed. "
                f"Please install: pip install 'sentence-transformers>={MIN_SENTENCE_TRANSFORMERS_VERSION}'"
            )

        required_version = _parse_version(MIN_TRANSFORMERS_VERSION_GEMMA)
        if current_version < required_version:
            warnings.warn(
                f"Gemma works best with transformers >= {MIN_TRANSFORMERS_VERSION_GEMMA}, "
                f"but you have {transformers_version}. Some features may not work correctly.",
                UserWarning
            )

        st_version = _get_package_version("sentence-transformers")
        st_current = _parse_version(st_version)
        st_required = _parse_version(MIN_SENTENCE_TRANSFORMERS_VERSION)

        if st_current < st_required:
            warnings.warn(
                f"sentence-transformers >= {MIN_SENTENCE_TRANSFORMERS_VERSION} is recommended, "
                f"but you have {st_version}. Consider upgrading: "
                f"pip install 'sentence-transformers>={MIN_SENTENCE_TRANSFORMERS_VERSION}'",
                UserWarning
            )

    return True, ""


class InstructorEmbeddingModel(ABC):
    """
    Base class for embedding models that support task-specific prompts.
    """

    def __init__(self, embed_model: str, model_type: str, task_prompts: Dict[str, str], eos_token: str = None):
        """
        Initialize the embedding model with task-specific prompts.
        """
        is_compatible, error_msg = _check_version_compatibility(model_type)
        if not is_compatible:
            raise ValueError(error_msg)

        self.embed_model = embed_model
        self.task_prompts = task_prompts
        self.task_id = None

    @abstractmethod
    def _encode_batch(self, formatted_batch: List[str]) -> torch.Tensor:
        """
        Model-specific encoding implementation.
        """
        pass

    def __call__(self, batch: List[str], batch_ids: Optional[List] = None):
        """
        Encode batch of texts into embeddings.
        """
        formatted_batch = []

        if type(self.task_id) != dict:
            # Non-search tasks
            prompt = self.task_prompts[self.task_id]
            formatted_batch = [f"{prompt}{text}" for text in batch]
        else:
            # Search task
            for i, (_, batch_type) in enumerate(batch_ids):
                if batch_type == 'q':
                    prompt = self.task_prompts['[SRCH]']['q']
                else:  # batch_type == 'c'
                    prompt = self.task_prompts['[SRCH]']['c']
                formatted_batch.append(f"{prompt}{batch[i]}")

        return self._encode_batch(formatted_batch)


class GemmaModel(InstructorEmbeddingModel):
    """
    Gemma embedding model using SentenceTransformer.

    Requirements:
        - transformers >= 4.56.0
        - sentence-transformers >= 2.7.0
    """

    def __init__(self, embed_model: str, task_prompts: Dict[str, str]):
        """
        Initialize Gemma embedding model.
        """
        super().__init__(embed_model, "gemma", task_prompts)

        self.encoder = SentenceTransformer(self.embed_model)
        self.tokenizer = self.encoder.tokenizer

        if hasattr(self.tokenizer, 'eos_token'):
            self.tokenizer.sep_token = self.tokenizer.eos_token

    def _encode_batch(self, formatted_batch: List[str]) -> torch.Tensor:
        return self.encoder.encode(formatted_batch, convert_to_tensor=True, device="cuda")

    def __call__(self, batch: List[str], batch_ids: Optional[List] = None):
        formatted_batch = []

        if type(self.task_id) != dict:
            # Non-search tasks
            prompt = self.task_prompts[self.task_id]
            formatted_batch = [prompt.format(**{'content':text}) for text in batch]
        else:
            # Search task
            for i, (_, batch_type) in enumerate(batch_ids):
                if batch_type == 'q':
                    prompt = self.task_prompts['[SRCH]']['q']
                    formatted_batch.append(prompt.format(**{'content':batch[i]}))
                else:  # batch_type == 'c'
                    prompt = self.task_prompts['[SRCH]']['c']
                    field_names = [field_name for _, field_name, _, _ in Formatter().parse(prompt) if field_name]
                    if 'title' in field_names:
                        title, text = batch[i].split(self.tokenizer.sep_token)
                        formatted_batch.append(prompt.format(**{'title': title.strip(), 'content': text.strip()}))
                    else:
                        formatted_batch.append(prompt.format(**{"content": batch[i]}))

        return self._encode_batch(formatted_batch)


class Qwen3Model(InstructorEmbeddingModel):
    """
    Qwen3 embedding model with last-token pooling.

    Requirements:
        - transformers >= 4.51.0
    """

    def __init__(self, embed_model: str, task_prompts: Dict[str, str]):
        """
        Initialize Qwen3 embedding model.
        """
        super().__init__(embed_model, "qwen3", task_prompts)

        self.encoder = SentenceTransformer(embed_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.embed_model, trust_remote_code=True)

        if hasattr(self.tokenizer, 'eos_token'):
            self.tokenizer.sep_token = self.tokenizer.eos_token

    def _encode_batch(self, formatted_batch: List[str]) -> torch.Tensor:
        """
        Encode batch using Qwen3 model with last-token pooling.

        Args:
            formatted_batch: Pre-formatted text strings

        Returns:
            Tensor of embeddings
        """
        return self.encoder.encode(sentences=formatted_batch, convert_to_tensor=True, device="cuda")
    
    def __call__(self, batch: List[str], batch_ids: Optional[List] = None):
        formatted_batch = []

        if type(self.task_id) != dict:
            # Non-search tasks
            prompt = self.task_prompts[self.task_id]
            formatted_batch = [prompt.format(**{'content':text}) for text in batch]
        else:
            # Search task
            for i, (_, batch_type) in enumerate(batch_ids):
                prompt = self.task_prompts['[SRCH]'][batch_type]
                formatted_batch.append(prompt.format(**{'content':batch[i]}))

        return self._encode_batch(formatted_batch)
