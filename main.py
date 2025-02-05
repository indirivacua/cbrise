from cbrise.mask_generator import MaskGenerator, RandomMaskGenerator
from cbrise.perturbation import ConstantPerturbation, BlurPerturbation, Perturbation
from cbrise.stopping_criteria import MaxIterations, NoImprovement, StoppingCriteria

from cbrise.rise import RISE

import torch
from torch import nn
from torchvision import models, transforms

import requests
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from time import perf_counter

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

batch_size = 32

mask_generator = RandomMaskGenerator(batch_size, (4, 4), (224, 224))
stopping_criteria = NoImprovement(
    1000, 64 / batch_size, 0.3
)  # MaxIterations(128/batch_size)
isotropic_sigma = 10
reference = BlurPerturbation(
    torch.tensor([isotropic_sigma, isotropic_sigma])
)  # ConstantPerturbation(0)

rise = RISE(mask_generator, reference, stopping_criteria)

# plt.imshow(mask_generator.__next__(None).to("cpu")[0],cmap="gray")
# plt.show()

DEVICE, DTYPE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    torch.float32,
)

model = models.vgg16(weights="DEFAULT").to(device=DEVICE, dtype=DTYPE)
model = nn.Sequential(
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    model,
    nn.Softmax(dim=1),
)
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(device=DEVICE, dtype=DTYPE)),
    ]
)

img_url = "https://i.imgur.com/WuG5RVB.jpg"
response = requests.get(img_url, headers={"User-agent": ""})
img = Image.open(BytesIO(response.content))
input = transform(img)
input = input.unsqueeze(dim=0)  # Add batch dimension

output = model(input)
prediction_score, pred_label_idx = torch.topk(output, 1)

start_time = perf_counter()
metrics = {}


def plot_masked_input(i, masks, masked_inputs, scores, heatmap):
    plt.imshow(masked_inputs[0].permute(1, 2, 0).cpu())
    plt.show()


heatmap = rise.attribute(
    input, model, pred_label_idx, metrics=metrics
)  # ,callback=plot_masked_input)
metrics["iterations"] *= batch_size

print("Elapsed time: ", perf_counter() - start_time)
print("Metrics: ", metrics)
print("Heatmap shape: ", heatmap.shape)
print("Min and max heatmap values: ", heatmap.min(), heatmap.max())

plt.imshow(input[0].permute(1, 2, 0).cpu())
plt.imshow(heatmap[0, 0, :, :].cpu(), cmap="jet", alpha=0.5)
plt.colorbar()
plt.show()
