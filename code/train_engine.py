# Imports
import torch
from torch import nn
from tqdm.auto import tqdm

# Setup default device (for if device argument is not passed)
if torch.cuda.is_available():
  device = "cuda"
elif torch.mps.is_available():
  device = "mps"
else:
  device = "cpu"

# Create train_step(), test_step() and train() functions
# Functionize train and test step
def train_step(model: nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optim: torch.optim,
               device = device):
  """Performs the training step for an epoch

  Loops through the data_loader, performs a forward pass, a loss/acc calculation,
  zeros the gradients, performs backpropogation, and steps the optimizer
  Args:
    model: nn.Module: the model to be trained
    train_dataloader: the dataloader for the model to be trained on,
    optim: the optimizer to use
    device: the device to train the model on
      By default, goes to cuda, then mps, then cpu
  """
  train_loss = 0
  train_acc = 0
  model.train()

  for batch, (X,y) in enumerate(data_loader):
    X, y = X.to(device), y.to(device)
    y_logits = model(X)
    loss = loss_fn(y_logits, y)
    train_loss+=loss.item()
    optim.zero_grad()
    loss.backward()
    optim.step()

    y_pred_labels = y_logits.argmax(dim=1)
    train_acc += ((y_pred_labels == y).sum().item()/len(y_pred_labels))

  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  # print(f"Train loss: {train_loss}, Train acc: {train_acc}")
  return train_loss, train_acc

def test_step(model: nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              device=device):
  """Performs the testing step for an epoch

  Loops through the data_loader, performs a forward pass, a loss/acc calculation,
  zeros the gradients, performs backpropogation, and steps the optimizer
  Args:
    model: nn.Module: the model to be tested
    train_dataloader: the dataloader for the model to be tested on,
    device: the device to be used
      By default, goes to cuda, then mps, then cpu
  """

  test_loss = 0
  test_acc = 0

  model.eval()

  with torch.inference_mode():
    for batch, (X_test,y_test) in enumerate(data_loader):
      X_test,y_test = X_test.to(device), y_test.to(device)
      test_logits = model(X_test)
      loss = loss_fn(test_logits,y_test)
      test_loss+=loss.item()

      test_pred_labels = test_logits.argmax(dim=1)
      test_acc += ((test_pred_labels == y_test).sum().item()/len(test_pred_labels))


  test_loss /= len(data_loader)
  test_acc /= len(data_loader)
  # print(f"Test loss: {test_loss}, Test acc: {test_acc}")
  return test_loss, test_acc

def train_with_state_dict(model: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optim: torch.optim.Optimizer,
          loss_fn: nn.Module,
          epochs: int,
          device = device):
  """
  Trains a model.
  
  Trains a PyTorch model for a set number of epochs. Returns a 
  dictionary of some evaluation metrics of the model and the 
  state_dict of the model at the epoch with the highest test accuracy
  
  Args:
    train_dataloader: the dataloader to train the model on.
    test_dataloader: the dataloader to evaluate the model on.
    optim: the optimizer to use.
    loss_fn: the loss function to evaluate the model with.
    epochs: the number of epochs to train the model for.
    device: the device to set the model to.
  
  Returns:
    a tuple: (results, max_test_acc_dict)
      results: a dictionary full of the results of the model training.
      Includes:
        train_loss: loss for training data
        train_acc: accuracy for training data
        test_loss: loss for testing data
        test_acc: accuracy for testing data
      max_test_acc_dict: the state_dict of the model with the highest test_acc
  """



  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []}

  max_test_acc = 0
  max_test_acc_dict = {}

  for epoch in tqdm(range(epochs)):
    # print(f"\nEpoch: {epoch}")
    train_loss, train_acc = train_step(model=model,
             data_loader=train_dataloader,
             loss_fn=loss_fn,
             optim=optim,
             device=device)
    test_loss, test_acc = test_step(model=model,
            data_loader=test_dataloader,
            loss_fn=loss_fn)
    if test_acc > max_test_acc:
      max_test_acc = test_acc
      max_test_acc_dict = model.state_dict()

    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  return results, max_test_acc_dict