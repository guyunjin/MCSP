from .USNID import USNIDModel
from .MCN import MCNModel
from .CC import CCModel
from .SCCL import SCCLModel
from .MCSP import MCSPModel

methods_map = {
    'usnid': USNIDModel,
    'mcn': MCNModel,
    'cc': CCModel,
    'sccl': SCCLModel,
    'mcsp': MCSPModel,
}
