from .MCN import MCN
from .MCSP import MCSP
from .BERT_TEXT import BERT_TEXT
multimodal_methods_map = {
    'mcn': MCN,
    'mcsp': MCSP,
    'text': BERT_TEXT,
}