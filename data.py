import medmnist
from torchvision.transforms import transforms
import numpy as np
from medmnist import INFO, Evaluator

def medmnist_iid(dataset, num_users,data_points_per_user):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    if data_points_per_user >= len(dataset):
        print("Can't have the number of data points larger than the dataset")
    else:
        num_items = data_points_per_user
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def medmnist_dataset(name, num_clients,data_points_per_user):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    download = True
    data_flag = "pathmnist"
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])
    if name=="PathMnist":
        data_dir = "../../data/pathmnist"

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        
        # load the data      
        train_dataset = DataClass(root=data_dir,split='train', transform=apply_transform, download=download)
        test_dataset = DataClass(root=data_dir,split='test', transform=apply_transform, download=download)

        pil_dataset = DataClass(root=data_dir,split='train', download=download)
        # sample training data amongst users
        user_groups = medmnist_iid(train_dataset, num_clients,data_points_per_user)
        
    return train_dataset, test_dataset, user_groups, info