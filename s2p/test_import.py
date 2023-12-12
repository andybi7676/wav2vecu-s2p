from kaldi.base import set_verbose_level
from kaldi.decoder import (
    FasterDecoder,
    FasterDecoderOptions,
    LatticeFasterDecoder,
    LatticeFasterDecoderOptions,
)
from kaldi.lat.functions import DeterminizeLatticePhonePrunedOptions
from kaldi.fstext import read_fst_kaldi, SymbolTable
from kaldi.asr import FasterRecognizer, LatticeFasterRecognizer

print("yes")