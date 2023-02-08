import copy
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from models import MODEL
import numpy as np
from client import Client
from data import medmnist_dataset


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

if __name__=="__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLIENTS = 100
    EPOCHS = 10
    FRACTION = 0.2
    EVERY = 1
    LOCAL_EPOCHS = 10
    train_dataset, test_dataset, clients = medmnist_dataset(name="PathMnist", num_clients=NUM_CLIENTS)
    
    global_model = MODEL()
    global_model.to(DEVICE)
    global_model.train()
    local_weights = []
    local_losses = []
    train_loss = []
    training_accuracy = []
    for epoch in tqdm(range(EPOCHS)):
        local_weights, local_losses = [], []
        global_model.train()
        m = max(int(FRACTION*NUM_CLIENTS), 1) # number of clients for each round
        global_weights = global_model.state_dict()
        
        set_of_clients = np.random.choice(range(NUM_CLIENTS), m, replace=False)

        for client_idx in set_of_clients:
            local_model = Client(dataset=train_dataset, idxs=clients[client_idx], local_epochs=LOCAL_EPOCHS, device=DEVICE)
            weights , loss = local_model.update_weights(model=copy.deepcopy(global_model))
            local_weights.append(copy.deepcopy(weights)) 
            local_losses.append(copy.deepcopy(loss))
        
        # get the averaged weights
        global_weights = average_weights(local_weights)
        
        # update global model with averaged weights
        global_model.load_state_dict(global_weights)

        # average of the losses
        loss_avg = np.mean(local_losses)
        train_loss.append(loss_avg)

        # getting the average training accuracy over all clients at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for client in range(NUM_CLIENTS):
            local_model = Client(dataset=train_dataset, idxs=clients[client_idx])
            acc, loss = local_model.test_model(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        training_accuracy.append(np.mean(list_acc))

        if epoch % EVERY == 0:
            print(f"EPOCH {epoch + 1}:")
            print(f'Average training loss: {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*training_accuracy[-1]))
    
    # Testing how the model is doing after all the traingin rounds
    global_model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.NLLLoss().to(DEVICE)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Inference
        outputs = global_model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    print(f' \n Results after {EPOCHS} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*training_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*accuracy))