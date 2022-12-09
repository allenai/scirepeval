import logging
import warnings
from typing import Optional, List, Iterable, Tuple

import torch
from torch import nn
from transformers.adapters import ModelAdaptersMixin

from transformers.file_utils import add_start_docstrings
from transformers.models.opt.modeling_opt import OPT_START_DOCSTRING, OPTModel, OPTPreTrainedModel
from transformers.adapters.composition import adjust_tensors_for_parallel

logger = logging.getLogger(__name__)


@add_start_docstrings(
    """
The OPT Model that allows the loading of different heads dor different tasks. This enables a flexible use of the
models and adpters. Since this class does classification on the last token, it requires to know the position of the
last token. If a :obj:`pad_token_id` is defined in the configuration, it finds the last token that is not a padding
token in each row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each row of the batch. Since
it cannot guess the padding tokens when :obj:`inputs_embeds` are passed instead of :obj:`input_ids`, it does the same
(take the last value in each row of the batch).
""",
    OPT_START_DOCSTRING,
)
class OPTAdapterModel(ModelAdaptersMixin, OPTPreTrainedModel):
    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.model.decoder.layers):
            yield i, layer

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)

        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()
        self._init_head_modules()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds=None,
            use_cache: Optional[bool] = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            head=None,
            **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        batch_size = outputs[0].shape[0]

        if self.config.pad_token_id is None:
            # TODO-AH: this may result in unexpected behavior for classification. Find a better way to do this?
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                (sequence_lengths,) = adjust_tensors_for_parallel(outputs[0], sequence_lengths)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        cls_logits = outputs[0][range(batch_size), sequence_lengths]

        outputs = self.forward_head(
            outputs,
            head_name=head,
            cls_output=cls_logits,
            attention_mask=attention_mask,
            return_dict=return_dict,
            **kwargs,
        )

        return outputs
