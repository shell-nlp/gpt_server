from typing import List, Type, Union
from pydantic import BaseModel
from transformers.generation.logits_process import LogitsProcessor
from transformers import PreTrainedTokenizerBase
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    STOPPING_CRITERIA_INPUTS_DOCSTRING,
    add_start_docstrings,
)
import xgrammar as xgr
import torch


class XgrammarLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
        self.grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        # -----------

    def get_json_grammar_processor(self):
        compiled_grammar = self.grammar_compiler.compile_builtin_json_grammar()
        self.xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
        return self.xgr_logits_processor

    def get_json_schema_processor(self, schema: Union[str, Type[BaseModel]]):
        compiled_grammar = self.grammar_compiler.compile_json_schema(schema)
        self.xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
        return self.xgr_logits_processor

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        return self.xgr_logits_processor(input_ids=input_ids, scores=scores)


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class StopAtSpecificTokenCriteria(StoppingCriteria):
    """
    当生成出第一个指定token时，立即停止生成
    """

    def __init__(self, token_id_list: List[int] = None):
        """
        :param token_id_list: 停止生成的指定token的id的列表
        """
        self.token_id_list = token_id_list
        self.stop = False

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        # return np.argmax(scores[-1].detach().cpu().numpy()) in self.token_id_list
        # 储存scores会额外占用资源，所以直接用input_ids进行判断
        if self.stop:
            return True
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list
