from pathlib import Path
from sentencepiece import SentencePieceProcessor
from typing import List
from enum import Enum

class SpecialTokens(str, Enum):
    bos = "<s>"
    eos = "</s>"
    begin_inst = "[INST]"
    end_inst = "[/INST]"
    begin_tools = "[AVAILABLE_TOOLS]"
    end_tools = "[/AVAILABLE_TOOLS]"
    begin_tool_results = "[TOOL_RESULTS]"
    end_tool_results = "[/TOOL_RESULTS]"
    tool_calls = "[TOOL_CALLS]"
    prefix = "[PREFIX]"
    middle = "[MIDDLE]"
    suffix = "[SUFFIX]"

class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def n_words(self) -> int:
        return self._model.vocab_size()

    @property
    def bos_id(self) -> int:
        return self._model.bos_id()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str, bos: bool = True) -> List[int]:
        assert isinstance(s, str)
        t = self._model.encode(s)
        if bos:
            t = [self.bos_id, *t]
        return t

    def decode(self, t: List[int]) -> str:
        return self._model.decode(t)
