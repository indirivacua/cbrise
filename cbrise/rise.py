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
        while not self.stopping_criteria.stop(i,heatmap):
            masks = self.mask_generator.__next__(scores).to(input.device)
            masked_inputs = self.perturbation.perturb(input,masks)
            scores = model(masked_inputs)
            heatmap = self.update_heatmap(heatmap,masks,scores,target)
            if callback:
                callback(i,masks,masked_inputs,scores,heatmap)
            i+=1
        if metrics is not None:
            metrics['iterations'] = i
        return heatmap
    def update_heatmap(self,heatmap:torch.Tensor,masks:torch.Tensor,scores:torch.Tensor,target:int):
        scores = scores[:,target]
        scores = scores.view(-1,1,1,1) #broadcast
        return heatmap + torch.sum(masks*scores, dim=0, keepdim=True)

if __name__ == "__main__":

    from torchvision import models,transforms
    from PIL import Image
    from io import BytesIO
    import requests
    import matplotlib.pyplot as plt
    from time import perf_counter

    batch_size=32

    mask_generator = RandomMaskGenerator(batch_size,(4,4),(224,224))
    stopping_criteria = NoImprovement(1000,64/batch_size,0.3)#MaxIterations(128/32)
    isotropic_sigma = 10
    reference = BlurPerturbation(torch.tensor([isotropic_sigma,isotropic_sigma]))#ConstantPerturbation(0)

    rise = RISE(mask_generator,reference,stopping_criteria)

    # plt.imshow(mask_generator.__next__(None).to("cpu")[0],cmap="gray")
    # plt.show()

    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(device)
    # Add Softmax to last layer
    last_layer_name, last_layer = list(model._modules.items())[-1]
    model._modules[last_layer_name] = torch.nn.Sequential(
        last_layer,
        torch.nn.Softmax(dim=1),
    )
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_url="https://i.imgur.com/WuG5RVB.jpg"
    response = requests.get(img_url, headers = {'User-agent': ''})
    img = Image.open(BytesIO(response.content))
    input = transform(img)
    input = input.unsqueeze(dim=0) #add batch dim
    input = input.float()
    input = input.to(device)
    output = model(input)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    start_time = perf_counter()
    metrics={}
    def plot_masked_input(i,masks,masked_inputs,scores,heatmap):
        plt.imshow(masked_inputs[0].permute(1,2,0).cpu())
        plt.show()
    heatmap=rise.attribute(input,model,pred_label_idx,metrics=metrics)#,callback=plot_masked_input)
    metrics['iterations'] *= batch_size
    print("Elapsed time: ", perf_counter() - start_time)
    print("Metrics: ", metrics)
    print("Heatmap shape: ", heatmap.shape)
    print("Min and max heatmap values: ", heatmap.min(),heatmap.max())
    plt.imshow(input[0].permute(1,2,0).cpu())
    plt.imshow(heatmap[0,0,:,:].cpu(),cmap="jet",alpha=0.5)
    plt.colorbar()
    plt.show()