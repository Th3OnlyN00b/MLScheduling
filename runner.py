import torch as torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.nn as nn
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch.nn import Conv2d
from torchvision.datasets import CIFAR10, CelebA
from torchvision.transforms import ToTensor, Resize
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import json
from threading import Thread, Event, Lock
import sys
import concurrent.futures
from queue import Queue, PriorityQueue

if len(sys.argv) < 2:
    print("need the scheduler. Correct usage: `" + " ".join(sys.argv) + " [fifo|gpu|wsjfB|wsjfJ|wsjfI]" + "`")
    quit()

def runJob(job, device, continueRunning, event=None, q=None, batchSize=None):
    if(job["dataset"] == "cifar10"):
        test_dataset = CIFAR10(root='data/', download=True, train=False, transform=transforms.Compose([ToTensor()]))
        cifar10 = True
    else:
        test_dataset = CelebA(root='data/', download=True, transform=transforms.Compose([Resize((32, 32)), ToTensor()]))
        cifar10 = False
    
    batch_size = job["batchSize"] if batchSize == None else 8

    torch.manual_seed(43)

    test_loader = DataLoader(test_dataset, batch_size, num_workers=4, pin_memory=True)

    if(job["model"] == "resnet50"):
        model = models.resnet50(pretrained=True)
    elif(job["model"] == "resnet18"):
        model = models.resnet18(pretrained=True)
    elif(job["model"] == "squeezenet"):
        model = models.squeezenet1_0(pretrained=True)
    else:
        if(cifar10):
            model = models.googlenet(pretrained=True)
        else:
            model = models.googlenet(pretrained=False, num_classes=40)

    for param in model.parameters():
        param.requires_grad = False
    
    if cifar10:
        criterion = nn.NLLLoss()
    else:
        criterion = nn.NLLLoss()
        if(job["model"] == "squeezenet"):
            model.classifier._modules["1"] = nn.Conv2d(512, 40, kernel_size=(1, 1))
            model.num_classes = 40
        if(job["model"] != "googlenet"):
            model.fc = nn.Linear(2048, 40)
    goog = (job["model"] == "googlenet") and (not cifar10)
    model.to(device)
    print("job id:", job["id"])

    start = time.time()
    with torch.no_grad():
        avgCorrect = 0
        i = 0
        for inputs, labels in test_loader:
            # check to see if we need to pause execution
            if(continueRunning.is_set()):
                continueRunning.wait(10)
            continueRunning.clear()
            if(not cifar10) and (i > 50):
                break
            i += 1
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            if cifar10:
                batch_loss = criterion(logps, labels)
            else:
                if goog:
                    logps = logps.logits
                batch_loss = torch.zeros(1).to(device)

                for j in range(batch_size):
                    batch_loss += torch.sum(torch.abs(torch.sub(logps[j], labels[j])))
            
            ps = torch.exp(logps)
            _, top_class = ps.topk(1, dim=1)
            if cifar10:
                equals = top_class == labels.view(*top_class.shape)
            else:
                equals = top_class == torch.argmax(labels, axis=1)
            avgCorrect += torch.mean(equals.type(torch.FloatTensor)).item()
    avgCorrect /= i
    job["finishTime"] = time.time()
    if event != None:
        q.put(job["size"])
        event.set()
    # We don't actually care, but the point is that we did actually test them all.
    print("Accuracy:", avgCorrect)

def addMoreJobs(pq, jobs, event):
    time.sleep(9)
    start = time.time()
    for job in jobs:
        job["start"] = start
        pq.put((job["priority"], job["size"], job["id"], job))
    print("======Added second round======")
    print(pq.qsize())
    event.set()

def removeCapacity(completedCaps, lock, used_cap):
    num = 0
    while True:
        cap = completedCaps.get()
        num += 1
        print(num)
        lock.acquire()
        used_cap -= cap
        lock.release()

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    with open("jobs.json", "r") as f:
        jobs = json.load(f)

    pq = PriorityQueue()
    secondRoundAdded = Event()
    Thread(target=addMoreJobs, args=(pq, jobs, secondRoundAdded), daemon=True).start()
    start = time.time()
    maxPrio = 0
    for job in jobs:
        job["start"] = start
        pq.put((job["priority"], job["size"], job["id"], job))
        if job["priority"] > maxPrio:
            maxPrio = job["priority"]

    MAX_CAPACITY = 5000
    capLock = Lock()
    used_capacity = 0
    completed_caps = Queue(len(jobs))
    tts = 0
    threads = []
    ev = Event()
    cache = []
    # Start the thread which will keep capacity up to date
    Thread(target=removeCapacity, args=(completed_caps, capLock, used_capacity), daemon=True).start()
    while(pq.qsize() > 0) or (not secondRoundAdded.is_set()) or (len(cache) > 0):
        # If we finished early and are just waiting for more jobs
        if(pq.qsize() == 0) and (len(cache) == 0):
            print("started2")
            secondRoundAdded.wait()
            print("ended2")
        # Bin packing
        if (sys.argv[1] != "fifo") and (sys.argv[1] != "gpu"):
            # Uncache any jobs we have cached
            for j in cache:
                pq.put((j["priority"], j["size"], j["id"], j))
            cache = []

            # Get a job to run
            job = pq.get()[3]
            # While we can keep loading in jobs
            capLock.acquire()
            while job["size"] + used_capacity <= MAX_CAPACITY:
                print("main entered2")
                used_capacity += job["size"]
                capLock.release()
                # Add the job and run it
                runEvent = Event()
                runEvent.clear()
                threads.append([Thread(target=runJob, args=(job, device, runEvent, ev, completed_caps, 1 if sys.argv[1] == "wsjfI" else None)), job, job["start"], runEvent])
                threads[-1][0].start()
                # Get the next job
                job = pq.get()[3]
                capLock.acquire()
            capLock.release()

            capLock.acquire()
            # While we have jobs too big to run
            while (job["size"] + used_capacity > MAX_CAPACITY):
                print("main entered1")
                # If this job is too big to be run on this system at all
                if(used_capacity == 0):
                    # Run it anyways, let the virtual paging take care of it
                    used_capacity += job["size"]
                    capLock.release()
                    runEvent = Event()
                    runEvent.clear()
                    threads.append([Thread(target=runJob, args=(job, device, runEvent, ev, completed_caps, 1 if sys.argv[1] == "wsjfI" else None)), job, start, runEvent])
                    threads[-1][0].start()
                    break
                capLock.release()

                
                if(sys.argv[1] == "wsjfB") or (sys.argv[1] == "wsjfI"):
                    # Find a thread to stop
                    print("stopping thread")
                    for thread in threads:
                        if thread[1]["priority"] < job["priority"]:
                            thread[3].clear()
                            break
                    break

                # If we have already cached all possible options:
                if pq.qsize() == 0:
                    # Wait for something to clear up
                    print("started1")
                    ev.wait()
                    print("ended1")
                    # Uncache the jobs
                    print(len(cache))
                    for j in cache:
                        pq.put((j["priority"], j["size"], j["id"], j))
                    cache = []
                    # Try to go legit
                    break
                # If neither of the above cases happened, cache this job and check the next if it fits
                cache.append(job)
                job = pq.get()[3]
                capLock.acquire()

            if capLock.locked():
                capLock.release()
            
            runEvent = Event()
            runEvent.clear()
            threads.append([Thread(target=runJob, args=(job, device, runEvent, ev, completed_caps, 1 if sys.argv[1] == "wsjfI" else None)), job, start, runEvent])
            threads[-1][0].start()
                    
        else: #fifo or GPU
            job = pq.get()[3]
            threads.append([Thread(target=runJob, args=(job, device)), job, start])
            threads[-1][0].start()
            if sys.argv[1] == "fifo":
                threads[-1][0].join()

    # Collect and analyze threads
    for thread in threads:
        thread[0].join()
        print(thread[1])
        tts += (thread[1]["finishTime"] - thread[2]) * (maxPrio - thread[1]["priority"])
    end = time.time()

    print("Final time:", end-start)
    print("Final tts:", tts)
