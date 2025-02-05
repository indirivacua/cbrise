from captum.attr import Attribution
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric

from typing import Callable, Optional, Dict

from .mask_generator import MaskGenerator
from .perturbation import Perturbation
from .stopping_criteria import StoppingCriteria
from .rise import RISE


class CaptumRISE(Attribution):
    def __init__(self, forward_func: Callable):
        super().__init__(forward_func)
        self.forward_func = forward_func

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        mask_generator: MaskGenerator,
        perturbation: Perturbation,
        stopping_criteria: StoppingCriteria,
        metrics: Optional[Dict] = None,
        callback: Optional[Callable] = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        rise = RISE(
            mask_generator=mask_generator,
            perturbation=perturbation,
            stopping_criteria=stopping_criteria,
        )

        attributions = rise.attribute(
            input=inputs,
            model=self.forward_func,
            target=target,
            metrics=metrics,
            callback=callback,
        )

        return attributions
