from typing import Iterable, Mapping, Optional

from torch.nn import Parameter, ParameterList, ParameterDict
from torchmeta.modules import MetaModule


class MetaParameterList(ParameterList, MetaModule):
    def __init__(self, parameters: Optional[Iterable['Parameter']] = None) -> None:
        super(MetaParameterList, self).__init__(parameters)


class MetaParameterDict(ParameterDict, MetaModule):
    def __init__(self, parameters: Optional[Mapping[str, 'Parameter']] = None) -> None:
        super(MetaParameterDict, self).__init__(parameters)
