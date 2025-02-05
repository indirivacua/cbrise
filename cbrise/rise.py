import torch
from typing import Callable

from .mask_generator import MaskGenerator
from .perturbation import Perturbation
from .stopping_criteria import StoppingCriteria

class RISE:
    def __init__(self,mask_generator:MaskGenerator,perturbation:Perturbation,stopping_criteria:StoppingCriteria) -> None:
        self.mask_generator=mask_generator
        self.perturbation=perturbation
        self.stopping_criteria=stopping_criteria
    @torch.no_grad()
    def attribute(self,input:torch.Tensor,model:torch.nn.Module,target:int,callback:Callable=None,metrics:dict=None):
        i = 0
        heatmap = torch.zeros_like(input).to(input.device)
        heatmap = heatmap[:,0,:].unsqueeze(1)
        scores = torch.zeros((0)).to(input.device)
        masks_sum = torch.zeros(input.shape[-2:]).to(input.device)
        original_scores = model(input)
        while not self.stopping_criteria.stop(i,heatmap):
            masks = self.mask_generator.__next__(scores).to(input.device)
            masked_inputs = self.perturbation.perturb(input,masks)
            masks_sum += masks.sum(dim=0).squeeze()
            scores = model(masked_inputs)
            heatmap = self.update_heatmap(heatmap,masks,original_scores,scores,target)
            if callback:
                callback(i,masks,masked_inputs,scores,heatmap)
            i+=1
        if metrics is not None:
            metrics['iterations'] = i
        normalized_heatmap = heatmap/masks_sum
        return normalized_heatmap
    def update_heatmap(self,heatmap:torch.Tensor,masks:torch.Tensor,original_scores:torch.Tensor,scores:torch.Tensor,target:int):
        prediction_score = original_scores[:,target]
        scores = scores[:,target]
        scores = scores.view(-1,1,1,1) #broadcast
        factor = torch.min(scores,prediction_score)/prediction_score
        return heatmap + torch.sum(masks*factor, dim=0, keepdim=True)