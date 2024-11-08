import sys
import pathlib

root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import json, re
from hardware_model.compute_module import (
    VectorUnit,
    SystolicArray,
    Core,
    ComputeModule,
    overhead_dict,
)
from hardware_model.io_module import IOModule
from hardware_model.memory_module import MemoryModule
from hardware_model.device import Device
from hardware_model.interconnect import LinkModule, InterConnectModule, TopologyType
from hardware_model.system import System
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.softmax import Softmax
from software_model.matmul import Matmul
from software_model.gelu import Gelu
from software_model.layernorm import LayerNorm
# from software_model.operators import

from software_model.utils import data_type_dict, Tensor
from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2
from math import ceil

from multiprocessing import Process, Lock
import time
from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2
