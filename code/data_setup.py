#Imports
import torch
import torchvision
def create_dataloaders(train_dir,
                      test_dir,
                      train_transform,
                      test_transform, 
                      batch_size,
                      num_workers,
                      device):
  """Creates training and test dataloaders for training a model.
  
  Uses training and test directories to turn them into datasets and the DataLoaders. Training and testing data should be in Standard Image Classsification Format. 
  Example:
  \ data
    \ train
      \ pizza
        \ image_1.jpg
        \ image_2.jpg
        ...
      \ steak
        \image_3.jpg
        \image_4.jpg
        ...
      \sushi
        \image_5.jpg
        \image_6.jpg
    \ test
      \ pizza
        \ image_1.jpg
        \ image_2.jpg
        ...
      \ steak
        \ image_3.jpg
        \ image_4.jpg
        ...
      \ sushi
        \ image_5.jpg
        \ image_6.jpg
  
  Args:
    train_dir: path to training directory,
    test_dir: path to testing directory,
    train_transform: the transform that should be applied to training data,
    test_transform: the transform that should be applied to testing data. If set to None, will be set to train_transform,
    batch size: number of images in each batch,
    num_workers: the number of workers to be used.

  Returns:
    A tuple (train_dataloader, test_dataloader)
      train_dataloader: a torch.utils.data.DataLoader instance using training images
      test_dataloader a torch.utils.data.DataLoader instance using testing images
  """
  # Create train and test datasets
  train_data = torchvision.datasets.ImageFolder(root=train_dir,
                                                transform=train_transform)
  test_data = torchvision.datasets.ImageFolder(root=test_dir,
                                               transform=test_transform)
  
  # Create train and test DataLoaders
  train_dataloader = torch.utils.data.DataLoader(dataset=train_data,
                                                 batch_size=batch_size,
                                                 shuffle=True)
  test_dataloader = torch.utils.data.DataLoader(dataset=test_data,
                                                batch_size=batch_size,
                                                shuffle=False)
  return train_dataloader, test_dataloader