from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Optional
import importlib.metadata
import warnings

# Lazy import for optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

instr_format = "Represent the Science document"

# Version requirements
MIN_TRANSFORMERS_VERSION_QWEN3 = "4.51.0"
MIN_TRANSFORMERS_VERSION_GEMMA = "4.56.0"
MIN_SENTENCE_TRANSFORMERS_VERSION = "2.7.0"


def _parse_version(version_str: str) -> tuple:
    """Parse version string into tuple of integers for comparison."""
    try:
        # Handle preview versions like "4.56.0-Embedding-Gemma-preview"
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

        # Check sentence-transformers version
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


def _last_token_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Perform last-token pooling on token embeddings (for Qwen3).
    Extracts the last non-padding token's hidden state.

    Args:
        last_hidden_states: Token-level embeddings from model output
        attention_mask: Attention mask

    Returns:
        Sentence-level embeddings using last token
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class GemmaModel:
    """
    Gemma embedding model using SentenceTransformer.

    Requirements:
        - transformers >= 4.56.0
        - sentence-transformers >= 2.7.0
    """

    def __init__(self, embed_model: str):
        """
        Initialize Gemma embedding model.

        Args:
            embed_model: Model identifier (e.g., "google/embeddinggemma-300m")

        Raises:
            ValueError: If version requirements are not met
        """
        # Check version compatibility
        is_compatible, error_msg = _check_version_compatibility("gemma")
        if not is_compatible:
            raise ValueError(error_msg)

        self.embed_model = embed_model
        self.task_id = None
        self.encoder = SentenceTransformer(self.embed_model)
        self.tokenizer = self.encoder.tokenizer

        # Gemma-specific instruction format (task descriptions for the "task:" field)
        self.instruction_map = {
            "[CLF]": "classification",
            "[RGN]": "regression",
            "[PRX]": "retrieval",
            "[SRCH]": {
                "q": "search result",  # Default task for queries
                "c": "retrieval"       # Task for documents/corpus
            }
        }

        if hasattr(self.tokenizer, 'eos_token'):
            self.tokenizer.sep_token = self.tokenizer.eos_token

    def __call__(self, batch: List[str], batch_ids: Optional[List] = None):
        """
        Encode batch of texts into embeddings.

        Args:
            batch: List of text strings to encode
            batch_ids: Optional batch identifiers for search tasks

        Returns:
            Tensor of embeddings
        """
        if type(self.task_id) != dict:
            # Non-search tasks: treat all as documents with task prefix
            task = self.instruction_map[self.task_id]
            # Format: "task: {task} | text: {content}"
            formatted_batch = [f"task: {task} | text: {text}" for text in batch]
            embeddings = self.encoder.encode(formatted_batch, convert_to_tensor=True)
        else:
            # Search task: distinguish between queries ('q') and documents ('c')
            embeddings_list = []
            for text, (_, batch_type) in zip(batch, batch_ids):
                if batch_type == 'q':
                    # Query format: "task: search result | query: {text}"
                    task = self.instruction_map['[SRCH]']['q']
                    formatted = f"task: {task} | query: {text}"
                    emb = self.encoder.encode(formatted, convert_to_tensor=True)
                else:  # batch_type == 'c' (corpus/document)
                    # Document format: "title: none | text: {content}"
                    formatted = f"title: none | text: {text}"
                    emb = self.encoder.encode(formatted, convert_to_tensor=True)
                embeddings_list.append(emb)
            embeddings = torch.stack(embeddings_list)

        return embeddings


class Qwen3Model:
    """
    Qwen3 embedding model with last-token pooling.

    Requirements:
        - transformers >= 4.51.0
    """

    def __init__(self, embed_model: str):
        """
        Initialize Qwen3 embedding model.

        Args:
            embed_model: Model identifier (e.g., "Qwen/Qwen3-Embedding-8B")

        Raises:
            ValueError: If version requirements are not met
        """
        # Check version compatibility
        is_compatible, error_msg = _check_version_compatibility("qwen3")
        if not is_compatible:
            raise ValueError(error_msg)

        self.embed_model = embed_model
        self.task_id = None
        self.encoder = AutoModel.from_pretrained(self.embed_model, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.embed_model, trust_remote_code=True)

        # Qwen3-specific instruction format
        # Note: For search tasks, only queries ('q') get instructions; documents ('c') are encoded directly
        self.instruction_map = {
            "[CLF]": f"Instruct: {instr_format} for classification\nQuery: ",
            "[RGN]": f"Instruct: {instr_format} for regression\nQuery: ",
            "[PRX]": f"Instruct: {instr_format} for retrieval\nQuery: ",
            "[SRCH]": {
                "q": "Instruct: Represent the Science query for retrieving supporting documents\nQuery: ",
                "c": ""  # Documents are encoded without instructions
            }
        }

        if hasattr(self.tokenizer, 'eos_token'):
            self.tokenizer.sep_token = self.tokenizer.eos_token

    def __call__(self, batch: List[str], batch_ids: Optional[List] = None):
        """
        Encode batch of texts into embeddings using last-token pooling.

        Args:
            batch: List of text strings to encode
            batch_ids: Optional batch identifiers for search tasks

        Returns:
            Tensor of embeddings
        """
        if type(self.task_id) != dict:
            # Non-search tasks: apply instruction format to all inputs
            formatted_batch = [f"{self.instruction_map[self.task_id]}{text}" for text in batch]
        else:
            # Search task: only apply instructions to queries ('q'), not documents ('c')
            formatted_batch = []
            for i, (_, batch_type) in enumerate(batch_ids):
                if batch_type == 'q':
                    # Query: use "Instruct: ... \nQuery: " format
                    formatted_batch.append(f"{self.instruction_map['[SRCH]']['q']}{batch[i]}")
                else:  # batch_type == 'c' (corpus/document)
                    # Document: no instruction, encode directly
                    formatted_batch.append(batch[i])

        # Tokenize and encode
        inputs = self.tokenizer(formatted_batch, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Use last-token pooling (critical for Qwen3)
            embeddings = _last_token_pooling(outputs.last_hidden_state, inputs['attention_mask'])

        return embeddings
