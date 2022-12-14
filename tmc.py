import os
import numpy as np
import pickle as pkl
from tqdm.notebook import tqdm

import torch
from torch.utils.data import RandomSampler, DataLoader
from .utils import accuracy, error

device = ('cuda' if torch.cuda.is_available() else 'cpu')


class DShap(object):

    def __init__(self, model, train_dataset, test_dataset,
                 directory=None, seed=10):
        """
        Args:
            model: Torch model
            train_dataset: Training Dataset (torch.Dataset)
            test_dataset: Test Dataset (torch.Dataset)
            directory: Directory to save results and figures.
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting
                same permutations.
        """

        if seed is not None:
            np.random.seed(seed)

        self.directory = directory
        if self.directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.model = model
        self.train_set = train_dataset
        self.test_set = test_dataset
        self.train_len = len(self.train_set)

        self.mem_tmc = np.zeros((0, self.train_len))
        self.idxs_tmc = np.zeros((0, self.train_len), int)
        test_classes = torch.tensor([label for _, label in self.test_set])
        self.random_score = torch.max(torch.bincount(test_classes) / len(self.test_set)).item()

        self.tmc_number = self._which_parallel(self.directory)
        self._create_results_placeholder(self.directory, self.tmc_number)

    def _create_results_placeholder(self, directory, tmc_number):
        tmc_dir = os.path.join(
            directory,
            'mem_tmc_{}.pkl'.format(tmc_number.zfill(4))
        )
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc},
                 open(tmc_dir, 'wb'))

    def run(self, save_every, err, tolerance=0.01):
        """Calculates data sources(points) values.

        Args:
            save_every: save marginal contributions every n iterations.
            err: stopping criteria.
            tolerance: Truncation tolerance. If None, it's computed.
        """

        # self.results = {}
        tmc_run = True
        while tmc_run:
            if error(self.mem_tmc) < err:
                tmc_run = False
            else:
                self.tmc_shap(
                    save_every,
                    tolerance=tolerance,
                )
                self.vals_tmc = np.mean(self.mem_tmc, 0)
            self.save_results()

    def save_results(self):
        """Saves results computed so far."""
        if self.directory is None:
            return
        tmc_dir = os.path.join(
            self.directory,
            'mem_tmc_{}.pkl'.format(self.tmc_number.zfill(4))
        )

        raw_list = []
        for j in range(self.mem_tmc.shape[0]):
            row_dict = {self.idxs_tmc[j][i]: self.mem_tmc[j][i] for i in range(self.mem_tmc.shape[1])}
            raw_list.append(row_dict)

        pkl.dump(raw_list, open(tmc_dir, 'wb'))
        # pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc},
        #         open(tmc_dir, 'wb'))

    def _which_parallel(self, directory):
        '''Prevent conflict with parallel runs.'''
        previous_results = os.listdir(directory)
        tmc_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                     for name in previous_results if 'mem_tmc' in name]
        tmc_number = str(np.max(tmc_nmbrs) + 1) if len(tmc_nmbrs) else '0'
        return tmc_number

    def tmc_shap(self, iterations, tolerance=0.01):
        """Runs TMC-Shapley algorithm.

        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance ratio.
        """
        self._tol_mean_score()

        marginals, idxs = [], []
        for _ in tqdm(range(iterations)):
            marginals, idxs = self.one_iteration(
                tolerance=tolerance
            )
            self.mem_tmc = np.concatenate([
                self.mem_tmc,
                np.reshape(marginals, (1, -1))  # dims: (some number, # train samples)
            ])
            self.idxs_tmc = np.concatenate([
                self.idxs_tmc,
                np.reshape(idxs, (1, -1))
            ])

    def one_iteration(self, tolerance):
        """Runs one iteration of TMC-Shapley algorithm."""
        idxs = np.random.permutation(
            self.train_len)  # Reread algorithm. We can get random sampler with a dataloader instead
        marginal_contribs = np.zeros(self.train_len)

        truncation_counter = 0
        new_score = self.random_score
        self.model.train()

        #  Iterates through the entire Training dataset
        data_list = []
        label_list = []
        for i, idx in enumerate(idxs):
            old_score = new_score
            data_list.append(self.train_set[idx][0])
            label_list.append(torch.tensor(self.train_set[idx][1]))
            if i == 0:
                data = self.train_set[idx][0].unsqueeze(0)
                labels = torch.tensor([self.train_set[idx][1]])
            else:
                data = torch.stack(data_list, 0)
                labels = torch.stack(label_list, 0)

            data, labels = data.to(device), labels.to(device)
            new_score = accuracy(self.model(data), labels)

            marginal_contribs[idx] = (new_score - old_score)  # original code divides by 1 for some reason
            distance_to_full_score = np.abs(new_score - self.mean_score)
            #  Performance Tolerance
            if distance_to_full_score <= tolerance * self.mean_score:
                truncation_counter += 1
                if truncation_counter > 5:
                    break
            else:
                truncation_counter = 0
        return marginal_contribs, idxs

    def _tol_mean_score(self):
        """Computes the average performance and its error using bagging."""
        scores = []
        self.model.eval()
        for _ in range(100):
            # bag_idxs = np.random.choice(len(self.y_test), len(self.y_test))  # check size

            sampler = RandomSampler(self.test_set)
            loader = DataLoader(self.test_set, batch_size=512, num_workers=2, sampler=sampler)

            # 1-pass
            for data, labels in loader:
                data, labels = data.to(device), labels.to(device)
                acc = accuracy(self.model(data), labels)
                scores.append(acc)
                break

        self.tol = np.std(scores)
        self.mean_score = np.mean(scores)