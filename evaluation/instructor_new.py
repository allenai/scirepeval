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

try:
    from gritlm import GritLM
    GRITLM_AVAILABLE = True
except ImportError:
    GRITLM_AVAILABLE = False

# Version requirements
MIN_TRANSFORMERS_VERSION_QWEN3 = "4.51.0"
MIN_TRANSFORMERS_VERSION_GEMMA = "4.56.0"
MIN_TRANSFORMERS_VERSION_GRITLM = "4.51.0"
MIN_SENTENCE_TRANSFORMERS_VERSION = "2.7.0"

# Task type constants
SEARCH_TASK_ID = '[SRCH]'
QUERY_TYPE = 'q'
CANDIDATE_TYPE = 'c'

# Field name constants
TITLE_FIELD = 'title'
CONTENT_FIELD = 'content'

# Special token constants
BERT_STYLE_SEP_TOKEN = '[SEP]'


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


class PromptFormatter:

    def __init__(self, task_prompts: Dict[str, str], tokenizer=None):
        self.task_prompts = task_prompts
        self.tokenizer = tokenizer

    def _parse_title_content(self, text: str, sep_token: str) -> tuple:
        parts = text.split(sep_token)
        if len(parts) >= 2:
            return parts[0].strip(), parts[1].strip()
        return parts[0].strip(), ""

    def _get_template_fields(self, prompt: str) -> List[str]:
        return [field_name for _, field_name, _, _ in Formatter().parse(prompt) if field_name]

    def _format_with_fields(self, prompt: str, text: str, sep_token: str = None) -> str:
        field_names = self._get_template_fields(prompt)

        if TITLE_FIELD in field_names and sep_token:
            title, content = self._parse_title_content(text, sep_token)
            return prompt.format(**{TITLE_FIELD: title, CONTENT_FIELD: content})
        else:
            return prompt.format(**{CONTENT_FIELD: text})

    def format_batch(self, batch: List[str], task_id, task_name: str = None,
                     batch_ids: Optional[List] = None, sep_token: str = None,
                     use_field_formatting: bool = True) -> List[str]:
        formatted_batch = []
        is_search_task = isinstance(task_id, dict)

        if not is_search_task:
            prompt = self.task_prompts[task_name] if task_name else self.task_prompts[task_id]

            if use_field_formatting:
                formatted_batch = [self._format_with_fields(prompt, text, sep_token) for text in batch]
            else:
                formatted_batch = [f"{prompt}{text}" for text in batch]
        else:
            for i, (_, batch_type) in enumerate(batch_ids):
                if task_name:
                    prompt = self.task_prompts[task_name][batch_type]
                else:
                    prompt = self.task_prompts[SEARCH_TASK_ID][batch_type]

                if use_field_formatting:
                    formatted_batch.append(self._format_with_fields(prompt, batch[i], sep_token))
                else:
                    formatted_batch.append(prompt.format(**{CONTENT_FIELD: batch[i]}))

        return formatted_batch


class InstructorEmbeddingModel(ABC):

    def __init__(self, embed_model: str, model_type: str, task_prompts: Dict[str, str], eos_token: str = None):
        is_compatible, error_msg = _check_version_compatibility(model_type)
        if not is_compatible:
            raise ValueError(error_msg)

        self.embed_model = embed_model
        self.task_prompts = task_prompts
        self.task_id = None
        self.task_name = None
        self.formatter = None

    def _setup_tokenizer_sep_token(self, tokenizer):
        if hasattr(tokenizer, 'eos_token'):
            tokenizer.sep_token = tokenizer.eos_token

    def _replace_sep_placeholder(self, batch: List[str]) -> List[str]:
        if not hasattr(self, 'tokenizer') or not hasattr(self.tokenizer, 'sep_token'):
            return batch

        sep_token = self.tokenizer.sep_token
        if sep_token == BERT_STYLE_SEP_TOKEN:
            return batch

        return [text.replace(BERT_STYLE_SEP_TOKEN, sep_token)
                if BERT_STYLE_SEP_TOKEN in text else text
                for text in batch]

    @abstractmethod
    def _encode_batch(self, formatted_batch: List[str]) -> torch.Tensor:
        pass

    def _get_sep_token(self) -> str:
        if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'sep_token'):
            return self.tokenizer.sep_token
        return None


class GemmaModel(InstructorEmbeddingModel):

    def __init__(self, embed_model: str, task_prompts: Dict[str, str]):
        super().__init__(embed_model, "gemma", task_prompts)

        self.encoder = SentenceTransformer(self.embed_model)
        self.tokenizer = self.encoder.tokenizer
        self._setup_tokenizer_sep_token(self.tokenizer)
        self.formatter = PromptFormatter(task_prompts, self.tokenizer)

    def _encode_batch(self, formatted_batch: List[str]) -> torch.Tensor:
        return self.encoder.encode(formatted_batch, convert_to_tensor=True, device="cuda")

    def __call__(self, batch: List[str], batch_ids: Optional[List] = None):
        formatted_batch = self.formatter.format_batch(
            batch=batch,
            task_id=self.task_id,
            task_name=self.task_name,
            batch_ids=batch_ids,
            sep_token=self._get_sep_token(),
            use_field_formatting=True
        )
        return self._encode_batch(formatted_batch)


class Qwen3Model(InstructorEmbeddingModel):

    def __init__(self, embed_model: str, task_prompts: Dict[str, str]):
        super().__init__(embed_model, "qwen3", task_prompts)

        self.encoder = SentenceTransformer(embed_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.embed_model)
        self._setup_tokenizer_sep_token(self.tokenizer)
        self.formatter = PromptFormatter(task_prompts, self.tokenizer)

    def _encode_batch(self, formatted_batch: List[str]) -> torch.Tensor:
        return self.encoder.encode(sentences=formatted_batch, convert_to_tensor=True, device="cuda")

    def __call__(self, batch: List[str], batch_ids: Optional[List] = None):
        batch = self._replace_sep_placeholder(batch)

        formatted_batch = self.formatter.format_batch(
            batch=batch,
            task_id=self.task_id,
            task_name=self.task_name,
            batch_ids=batch_ids,
            sep_token=self._get_sep_token(),
            use_field_formatting=True
        )
        return self._encode_batch(formatted_batch)


class GritLMModel(InstructorEmbeddingModel):

    def __init__(self, embed_model: str, task_prompts: Dict[str, str]):
        super().__init__(embed_model, "gritlm", task_prompts)

        if not GRITLM_AVAILABLE:
            raise ImportError(
                "GritLM requires the gritlm package. "
                "Please install: pip install gritlm"
            )

        self.encoder = GritLM(self.embed_model, torch_dtype="auto", mode="embedding")
        self.tokenizer = self.encoder.tokenizer
        self._setup_tokenizer_sep_token(self.tokenizer)

    @staticmethod
    def _gritlm_instruction(instruction: str) -> str:
        if instruction:
            return f"<|user|>\n{instruction}\n<|embed|>\n"
        else:
            return "<|embed|>\n"

    def _encode_batch(self, formatted_batch: List[str]) -> torch.Tensor:
        embeddings = self.encoder.encode(formatted_batch, convert_to_tensor=True)
        return embeddings

    def __call__(self, batch: List[str], batch_ids: Optional[List] = None):
        is_search_task = isinstance(self.task_id, dict)

        if not is_search_task:
            instruction = self.task_prompts.get(self.task_id, "")
            return self.encoder.encode(batch, instruction=self._gritlm_instruction(instruction), convert_to_tensor=True)
        else:
            query_instruction = self._gritlm_instruction(self.task_prompts[SEARCH_TASK_ID].get(QUERY_TYPE, ''))
            candidate_instruction = self._gritlm_instruction("")

            instructions = [
                query_instruction if batch_type == QUERY_TYPE else candidate_instruction
                for _, batch_type in batch_ids
            ]

            return self.encoder.encode(batch, instruction=instructions, convert_to_tensor=True)
