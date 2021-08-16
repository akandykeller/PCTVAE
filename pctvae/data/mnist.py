import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np
from functools import partial
from pctvae.data.transforms import AddPerspectiveTransformationDims, AddRandomTransformationDims, AddFixedTransformationDims, AddDualTransformationDims, To_Color
from pctvae.data.preprocessing import zca_whitening_matrix, NormalizePixels, StatsRecorder, ZCATransformation

def string_to_list(string):
    l = list(map(float, str(string).split(' ')))
    return l


class DuplicateTargets(object):
    def __init__(self, n_transforms):
        self.n_transforms = n_transforms

    def __call__(self, target):
        """
        Args:
            tensor (Tensor): Target tensor (N,)
        Returns:
            Tensor: Target tensor duplicated to the number of transformations (N * n_transforms)
        """
        return torch.tensor(target).view(-1, 1).repeat(1, self.n_transforms).view(-1,)

    def __repr__(self):
        format_string = self.__class__.__name__ 
        return format_string


class Preprocessor:
    def __init__(self, config):
        self.data_dir = config['data_dir']
        self.processed_zca_path = os.path.join(self.data_dir, 'zca_mnist.npy')
        self.processed_mean_path = os.path.join(self.data_dir, 'mean_mnist.npz')
        self.batch_size = config['batch_size']
        self.pct_train = config['pct_train']
        self.pct_val = config['pct_val']
        self.seed = config['seed']
        # self.whiten = config['whiten']
        # self.normalize = config['normalize']

        assert config['dataset'] == 'MNIST'
        self.dataset = getattr(datasets, config['dataset'])
        self.dataset_name = config['dataset']

        self.dataset_train = partial(self.dataset, train=True)
        self.dataset_test = partial(self.dataset, train=False)

        self.train_angle_set = string_to_list(config['train_angle_set'])
        self.test_angle_set = string_to_list(config['test_angle_set'])
        self.train_color_set = list(map(lambda x: np.radians(x) - np.pi, string_to_list(config['train_color_set'])))
        self.test_color_set = list(map(lambda x: np.radians(x) - np.pi, string_to_list(config['test_color_set'])))
        self.train_scale_set = string_to_list(config['train_scale_set'])
        self.test_scale_set = string_to_list(config['test_scale_set'])
        self.n_transforms_train = len(self.train_angle_set) * len(self.train_color_set) * len(self.train_scale_set)
        self.n_transforms_test = len(self.test_angle_set) * len(self.test_color_set) * len(self.test_scale_set)

        self.random_crop = config['random_crop']

        # Compute sampler for validation set
        self.get_train_val_split()
        # Compute/Load mean/std and whitenting for dataset transforms
        self.load_transform_params()

    def get_train_val_split(self, shuffle=True):
        data_train = self.dataset_train(self.data_dir, download=True)
        full_num_train = len(data_train)
        all_idxs = list(range(full_num_train))
        val_split = int(np.floor(self.pct_val * full_num_train))

        if shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(all_idxs)

        train_idxs, valid_idxs = all_idxs[val_split:], all_idxs[:val_split]

        train_subset_split = int(full_num_train * self.pct_train)
        train_idxs = train_idxs[:train_subset_split]

        self.train_sampler = SubsetRandomSampler(train_idxs)
        self.valid_sampler = SubsetRandomSampler(valid_idxs)

    def compute_zca_matrix(self):
        if self.whiten:
            if not os.path.exists(self.processed_zca_path):
                print("Can't find precomputed ZCA Matrix at {}".format(self.processed_zca_path))
                print("Computing data ZCA_Matrix")

                data_train = self.dataset_train(self.data_dir, transform=transforms.Compose(self.base_transforms_train), download=True)
                train_loader = DataLoader(data_train, batch_size=self.batch_size, drop_last=False, sampler=self.train_sampler)

                flat_processed_train_images = np.array([np.array(data_train[i][0]).reshape(-1) for i in range(len(data_train))])
                zca = zca_whitening_matrix(flat_processed_train_images.T)
                
                print("Saving whitening matrix at {}".format(self.processed_zca_path))
                np.save(self.processed_zca_path, zca)
            else:
                print("Loading whitening matrix from {}".format(self.processed_zca_path))
                zca = np.load(self.processed_zca_path)
            
            self.zca_mat = torch.from_numpy(zca).float()
        else:
            self.zca_mat = None
        return 


    def compute_mean_std(self):
        if not os.path.exists(self.processed_mean_path):
            print("Can't find Mean & Std at {}".format(self.processed_mean_path))
            print("Computing data Mean, STD")
            data_train = self.dataset_train(self.data_dir, transform=transforms.Compose(self.base_transforms_train), download=True)
            kwargs = {'num_workers': 20, 'pin_memory': True} if torch.cuda.is_available() else {}
            train_loader = DataLoader(data_train, batch_size=self.batch_size, drop_last=False, sampler=self.train_sampler, **kwargs)

            # Compute mean and std batchwise
            stats = StatsRecorder()
            for batch_idx, (data, target) in enumerate(train_loader):
                stats.update(data)
            mean = stats.mean
            std = stats.std
            # Replace zeros in std with 1.0
            std[std <= 1.e-5] = 1
            print("Saving Mean & Std at {}".format(self.processed_mean_path))
            try:
                np.savez(self.processed_mean_path, mean=mean, std=std)
            except:
                import pdb; pdb.set_trace()
        else:
            print("Loading Mean & Std from {}".format(self.processed_mean_path))
            loaded_stats = np.load(self.processed_mean_path)
            mean = loaded_stats['mean']
            std = loaded_stats['std']

        self.mean_val = torch.from_numpy(mean)
        self.std_val = torch.from_numpy(std)
        return


    def denormalize(self, data):
        std = self.std_val
        data = (data * std.to(data.device)) + self.mean_val.to(data.device)
        return data.clamp(0,1)


    def load_transform_params(self):
        self.base_transforms_train = self.base_transforms_test = [
                                   transforms.RandomCrop(self.random_crop),
                                   transforms.ToTensor()]

        if self.dataset_name == 'MNIST' and len(self.train_color_set) > 1:
            self.base_transforms_train = [To_Color()] + self.base_transforms_train
            self.base_transforms_test = [To_Color()] + self.base_transforms_test

        self.base_transforms_train = self.base_transforms_train + [AddRandomTransformationDims(self.train_angle_set, self.train_color_set, self.train_scale_set)] 
        self.base_transforms_test =  self.base_transforms_test + [AddRandomTransformationDims(self.test_angle_set, self.test_color_set, self.test_scale_set)]

        # self.base_transforms_train = self.base_transforms_train + [AddFixedTransformationDims(self.train_angle_set, self.train_color_set, self.train_scale_set)] 
        # self.base_transforms_test =  self.base_transforms_test + [AddFixedTransformationDims(self.test_angle_set, self.test_color_set, self.test_scale_set)]

        # self.compute_mean_std()
        # if self.normalize:
        #     self.base_transforms_train = self.base_transforms_train + [NormalizePixels(self.mean_val, self.std_val)]
        #     self.base_transforms_test = self.base_transforms_test + [NormalizePixels(self.mean_val, self.std_val)]

        # self.compute_zca_matrix()
        # if self.whiten:
        #     self.base_transforms_train = self.base_transforms_train + [ZCATransformation(self.zca_mat)]
        #     self.base_transforms_test = self.base_transforms_test + [ZCATransformation(self.zca_mat)]

        self.transforms_train = transforms.Compose(self.base_transforms_train)
        self.target_transform_train = DuplicateTargets(self.n_transforms_train)
        self.transforms_test = transforms.Compose(self.base_transforms_test)
        self.target_transform_test = DuplicateTargets(self.n_transforms_test)


    def load_datasets(self):
        data_train = data_val = self.dataset_train(self.data_dir, transform=self.transforms_train, 
                                                          target_transform=self.target_transform_train, download=True)
        data_test = self.dataset_test(self.data_dir, transform=self.transforms_test,
                                                         target_transform=self.target_transform_test, download=True)
        return data_train, data_val, data_test


    def get_dataloaders(self, batch_size):
        data_train, data_val, data_test = self.load_datasets()

        kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_loader = DataLoader(data_train, batch_size=batch_size, 
                                       sampler=self.train_sampler,
                                       drop_last=True, **kwargs)
        val_loader = DataLoader(data_val, batch_size=batch_size, 
                                     sampler=self.valid_sampler,
                                     shuffle=False, drop_last=False, **kwargs)
        test_loader = DataLoader(data_test, batch_size=batch_size, 
                                      shuffle=False, drop_last=False, **kwargs)

        return train_loader, val_loader, test_loader



class DualTransformPreprocessor(Preprocessor):
    def load_transform_params(self):
        self.base_transforms_train = self.base_transforms_test = [
                                    transforms.RandomCrop(self.random_crop),
                                    transforms.ToTensor()]

        if self.dataset_name == 'MNIST' and len(self.train_color_set) > 1:
            self.base_transforms_train = [To_Color()] + self.base_transforms_train
            self.base_transforms_test = [To_Color()] + self.base_transforms_test

        self.base_transforms_train = self.base_transforms_train + [AddDualTransformationDims(self.train_angle_set, self.train_color_set, self.train_scale_set)] 
        self.base_transforms_test =  self.base_transforms_test + [AddDualTransformationDims(self.test_angle_set, self.test_color_set, self.test_scale_set)]

        self.transforms_train = transforms.Compose(self.base_transforms_train)
        self.target_transform_train = DuplicateTargets(self.n_transforms_train)
        self.transforms_test = transforms.Compose(self.base_transforms_test)
        self.target_transform_test = DuplicateTargets(self.n_transforms_test)


class PersepctivePreprocessor(Preprocessor):
    def load_transform_params(self):
        self.base_transforms_train = self.base_transforms_test = [
                                    transforms.RandomCrop(self.random_crop),
                                    transforms.ToTensor()]

        if self.dataset_name == 'MNIST' and len(self.train_color_set) > 1:
            self.base_transforms_train = [To_Color()] + self.base_transforms_train
            self.base_transforms_test = [To_Color()] + self.base_transforms_test

        self.base_transforms_train = self.base_transforms_train + [AddPerspectiveTransformationDims(self.train_angle_set, self.train_color_set, self.train_scale_set)] 
        self.base_transforms_test =  self.base_transforms_test + [AddPerspectiveTransformationDims(self.train_angle_set, self.train_color_set, self.train_scale_set)]

        self.transforms_train = transforms.Compose(self.base_transforms_train)
        self.target_transform_train = DuplicateTargets(self.n_transforms_train)
        self.transforms_test = transforms.Compose(self.base_transforms_test)
        self.target_transform_test = DuplicateTargets(self.n_transforms_test)

