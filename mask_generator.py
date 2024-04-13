import abc
import torch

class MaskGenerator(abc.ABC):
    def __init__(self,batch_size:int) -> None:
        self.batch_size=batch_size
    @abc.abstractmethod()
    def __next__(previous_mask_scores:torch.tensor):
        pass

class RandomMaskGenerator(MaskGenerator):
    def __init__(self,batch_size:int,base_size:tuple[int],upsample_size:tuple[int]) -> None:
        super().__init__(batch_size)
        self.base_size=base_size
        self.upsample_size=upsample_size
    def __next__(self,previous_mask_scores: torch.tensor):
        base_size_channels = (self.batch_size, 1,*self.base_size)
        base_masks = torch.rand(base_size_channels)
        masks = torch.nn.functional.interpolate(base_masks,self.upsample_size)
        return masks[:,0,:]