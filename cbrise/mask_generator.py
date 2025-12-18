import abc
import torch

class MaskGenerator(abc.ABC):
    def __init__(self,batch_size:int) -> None:
        self.batch_size=batch_size
    @abc.abstractmethod
    def __next__(previous_mask_scores:torch.Tensor):
        pass

class RandomMaskGenerator(MaskGenerator):
    def __init__(self,batch_size:int,base_size:tuple[int],upsample_size:tuple[int],upsample_mode:str="bilinear") -> None:
        super().__init__(batch_size)
        self.base_size=base_size
        self.upsample_size=upsample_size
		self.upsample_mode=upsample_mode
    def __next__(self,previous_mask_scores: torch.Tensor):
        base_size_channels = (self.batch_size, 1,*self.base_size)
        base_masks = torch.rand(*base_size_channels)
        base_masks = (base_masks > 0.5).float()
        masks = torch.nn.functional.interpolate(base_masks,size=self.upsample_size,mode=self.upsample_mode,align_corners=True)
        masks_min = masks.view(self.batch_size, -1).min(dim=1)[0].view(self.batch_size, 1, 1, 1)
        masks_max = masks.view(self.batch_size, -1).max(dim=1)[0].view(self.batch_size, 1, 1, 1)
        masks = (masks - masks_min) / (masks_max - masks_min + 1e-8)
        return masks
