from typing import List, Union
from pydantic import BaseModel, conlist


class InputData(BaseModel):
    data: List[conlist(Union[float, int], min_items=13, max_items=13)]
    col_names: conlist(str, min_items=13, max_items=13)


class OutputData(BaseModel):
    predicted: List[int]
