from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Optional, Dict, Union
from abc import ABC, abstractmethod
import importlib.metadata
import warnings
from string import Formatter
import json
from copy import deepcopy

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
MIN_TRANSFORMERS_VERSION_QWEN3 = (4, 51, 0)
MIN_TRANSFORMERS_VERSION_GEMMA = (4, 56, 0)
MIN_SENTENCE_TRANSFORMERS_VERSION = (2, 7, 0)

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
        if current_version < MIN_TRANSFORMERS_VERSION_QWEN3:
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

        if current_version < MIN_TRANSFORMERS_VERSION_GEMMA:
            warnings.warn(
                f"Gemma works best with transformers >= {MIN_TRANSFORMERS_VERSION_GEMMA}, "
                f"but you have {transformers_version}. Some features may not work correctly.",
                UserWarning
            )

        st_version = _get_package_version("sentence-transformers")
        st_current = _parse_version(st_version)

        if st_current < MIN_SENTENCE_TRANSFORMERS_VERSION:
            warnings.warn(
                f"sentence-transformers >= {MIN_SENTENCE_TRANSFORMERS_VERSION} is recommended, "
                f"but you have {st_version}. Consider upgrading: "
                f"pip install 'sentence-transformers>={MIN_SENTENCE_TRANSFORMERS_VERSION}'",
                UserWarning
            )

    return True, ""


def _merge_prompts(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two prompt dictionaries.
    Override values take precedence over base values.
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key == "base_prompt":
            # Skip the base_prompt reference in the merged result
            continue
        elif key == "parameters":
            # Merge parameters separately
            if "parameters" in result:
                result["parameters"].update(value)
            else:
                result["parameters"] = deepcopy(value)
        elif isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # Recursively merge nested dictionaries (like [SRCH])
            result[key] = _merge_prompts(result[key], value)
        else:
            # Override the value
            result[key] = deepcopy(value)

    return result


def load_prompts(prompts_data: Dict, prompt_name: str, _visited: Optional[set] = None) -> Dict:
    """
    Load and resolve prompts from the prompts configuration.

    Args:
        prompts_data: The full prompts dictionary loaded from JSON
        prompt_name: The name of the prompt configuration to load
        _visited: Internal parameter to track visited prompts and prevent cycles

    Returns:
        A fully resolved prompt dictionary with all base_prompt references resolved

    Raises:
        ValueError: If prompt_name doesn't exist or if circular reference is detected
    """
    if prompt_name not in prompts_data:
        raise ValueError(f"Prompt configuration '{prompt_name}' not found in prompts data")

    # Track visited prompts to detect circular references
    if _visited is None:
        _visited = set()

    if prompt_name in _visited:
        raise ValueError(f"Circular reference detected: {prompt_name} has already been visited")

    _visited.add(prompt_name)

    prompt_config = prompts_data[prompt_name]

    # If this config has a base_prompt, recursively resolve it first
    if "base_prompt" in prompt_config:
        base_prompt_name = prompt_config["base_prompt"]
        base_config = load_prompts(prompts_data, base_prompt_name, _visited.copy())
        # Merge the base config with the current config
        return _merge_prompts(base_config, prompt_config)
    else:
        # No base prompt, return a deep copy of the config
        return deepcopy(prompt_config)


def load_prompts_from_file(file_path: str, prompt_name: str) -> Dict:
    """
    Load and resolve prompts from a JSON file.

    Args:
        file_path: Path to the JSON file containing prompt configurations
        prompt_name: The name of the prompt configuration to load

    Returns:
        A fully resolved prompt dictionary
    """
    with open(file_path, 'r') as f:
        prompts_data = json.load(f)

    return load_prompts(prompts_data, prompt_name)


class PromptFormatter:

    def __init__(self, task_prompts: Dict[str, str]):
        self.task_prompts = task_prompts

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

    def format_batch(self, batch: List[str], task_id: Union[str, dict], task_name: str = None,
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
        self.formatter = PromptFormatter(task_prompts)

    def _encode_batch(self, formatted_batch: List[str]) -> torch.Tensor:
        return self.encoder.encode(formatted_batch, convert_to_tensor=True,device="cuda")

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
        self.formatter = PromptFormatter(task_prompts)

    def _encode_batch(self, formatted_batch: List[str]) -> torch.Tensor:
        return self.encoder.encode(sentences=formatted_batch, convert_to_tensor=True)#, device="cuda")

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


class F2LLMModel(InstructorEmbeddingModel):

    def __init__(self, embed_model: str, task_prompts: Dict[str, str]):
        super().__init__(embed_model, "f2llm", task_prompts)

        # Load model and tokenizer using transformers
        self.model = AutoModel.from_pretrained(embed_model).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model)
        self._setup_tokenizer_sep_token(self.tokenizer)
        self.formatter = PromptFormatter(task_prompts)

    def _encode_batch(self, formatted_batch: List[str]) -> torch.Tensor:
        """
        Encode a batch of sentences using F2LLM's custom encoding strategy.
        """
        # Append EOS token to each sentence
        sentences_with_eos = [s + self.tokenizer.eos_token for s in formatted_batch]

        # Tokenize
        tokenized_inputs = self.tokenizer(
            sentences_with_eos,
            padding=True,
            return_tensors='pt',
            add_special_tokens=False
        ).to("cuda")

        # Get last hidden state
        with torch.no_grad():
            last_hidden_state = self.model(**tokenized_inputs).last_hidden_state

        # Extract embeddings from the final token position (EOS position)
        eos_positions = tokenized_inputs.attention_mask.sum(dim=1) - 1
        embeddings = last_hidden_state[
            torch.arange(len(sentences_with_eos), device="cuda"),
            eos_positions
        ]

        # L2 normalization
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

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
