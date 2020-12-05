import json
import torch
import uuid
from torchvision import transforms, models
from torchvision.datasets import CIFAR10, CelebA
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader

# Local imports
from torchsummary import summary_string


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("jobs_rough.json", "r") as f:
    jobs = json.load(f)

results = []
for job in jobs:
    if(job["dataset"] == "cifar10"):
        test_dataset = CIFAR10(root='data/', download=True, train=False, transform=transforms.Compose([ToTensor()]))
    else:
        test_dataset = CelebA(root='data/', download=True, transform=transforms.Compose([ToTensor()]))
    
    batch_size = 128

    torch.manual_seed(43)

    test_loader = DataLoader(test_dataset, batch_size, num_workers=4, pin_memory=True)
    
    if(job["model"] == "resnet50"):
        model = models.resnet50(pretrained=True)
    elif(job["model"] == "resnet18"):
        model = models.resnet18(pretrained=True)
    elif(job["model"] == "squeezenet"):
        model = models.squeezenet1_0(pretrained=True)
    else:
        model = models.googlenet(pretrained=True)

    model.to(device)
    print(tuple(test_loader.dataset[0][0].shape))
    # quit()
    job["size"] = float(summary_string(model, batch_size=job["batchSize"], input_size=tuple(test_loader.dataset[0][0].shape))[2])
    job["id"] = str(uuid.uuid1())

    results.append(job)

with open("jobs.json", "w") as f:
    json.dump(results, f)
