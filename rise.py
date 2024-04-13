import torch
from typing import Callable

from . import MaskGenerator, RandomMaskGenerator,BlurPerturbation, Perturbation,NoImprovement, StoppingCriteria

class RISE:
    def __init__(self,mask_generator:MaskGenerator,perturbation:Perturbation,stopping_criteria:StoppingCriteria) -> None:
        self.mask_generator=mask_generator
        self.reference=perturbation
        self.stopping_criteria=stopping_criteria
    def attribute(self,input:torch.tensor,model:torch.nn.Module,callback:Callable):        
        i = 0
        heatmap = torch.zeros_like(input).to(input.device)
        scores = torch.zeros((0)).to(input.device)
        while not self.stopping_criteria.stop(i,heatmap):
            masks = self.mask_generator.__next__(scores).to(input.device)
            masked_inputs = self.perturbation.perturb(input,masks)
            scores = model(masked_inputs)
            heatmap = self.update_heatmap(masks,scores)
            callback(i,masks,masked_inputs,scores,heatmap)
        return heatmap
    def update_heatmap(self,heatmap:torch.tensor,masks:torch.tensor,scores:torch.tensor):
        return heatmap + masks*scores #PONELE! TODO

mask_generator = RandomMaskGenerator(32,(4,4),(224,224))
stopping_criteria = NoImprovement(1000,5,0.01)
isotropic_sigma = 2
reference = BlurPerturbation(torch.tensor([isotropic_sigma,isotropic_sigma]))

rise = RISE(mask_generator,reference,stopping_criteria)