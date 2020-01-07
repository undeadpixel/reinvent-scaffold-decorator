import random

import numpy as np
import scipy.stats as sps

import torch
import torch.utils.data as tud
import torch.nn.utils as tnnu

import models.dataset as md
import utils.tensorboard as utb
import utils.scaffold as usc


class Action:
    def __init__(self, logger=None):
        """
        (Abstract) Initializes an action.
        :param logger: An optional logger instance.
        """
        self.logger = logger

    def _log(self, level, msg, *args):
        """
        Logs a message with the class logger.
        :param level: Log level.
        :param msg: Message to log.
        :param *args: The arguments to escape.
        :return:
        """
        if self.logger:
            getattr(self.logger, level)(msg, *args)


class TrainModelPostEpochHook(Action):

    def __init__(self, logger=None):
        """
        Initializes a training hook that runs after every epoch.
        This hook enables to save the model, change LR, etc. during training.
        :return:
        """
        Action.__init__(self, logger)

    def run(self, model, training_set, epoch):  # pylint: disable=unused-argument
        """
        Performs the post-epoch hook. Notice that model should be modified in-place.
        :param model: Model instance trained up to that epoch.
        :param training_set: List of SMILES used as the training set.
        :param epoch: Epoch number (for logging purposes).
        :return: Boolean that indicates whether the training should continue or not.
        """
        return True  # simply does nothing...


class TrainModel(Action):

    def __init__(self, model, optimizer, training_sets, batch_size, clip_gradient,
                 epochs, post_epoch_hook=None, logger=None):
        """
        Initializes the training of an epoch.
        : param model: A model instance, not loaded in sampling mode.
        : param optimizer: The optimizer instance already initialized on the model.
        : param training_sets: An iterator with all the training sets (scaffold, decoration) pairs.
        : param batch_size: Batch size to use.
        : param clip_gradient: Clip the gradients after each backpropagation.
        : return:
        """
        Action.__init__(self, logger)

        self.model = model
        self.optimizer = optimizer
        self.training_sets = training_sets
        self.batch_size = batch_size
        self.epochs = epochs
        self.clip_gradient = clip_gradient

        if not post_epoch_hook:
            self.post_epoch_hook = TrainModelPostEpochHook(logger=self.logger)
        else:
            self.post_epoch_hook = post_epoch_hook

    def run(self):
        """
        Performs a training epoch with the parameters used in the constructor.
        :return: An iterator of (total_batches, epoch_iterator), where the epoch iterator
                  returns the loss function at each batch in the epoch.
        """
        for epoch, training_set in zip(range(1, self.epochs + 1), self.training_sets):
            dataloader = self._initialize_dataloader(training_set)
            self._log("info", "Epoch dataset size: %d", len(dataloader))
            epoch_iterator = self._epoch_iterator(dataloader)
            yield len(dataloader), epoch_iterator

            self.model.set_mode("eval")
            post_epoch_status = self.post_epoch_hook.run(self.model, training_set, epoch)
            self.model.set_mode("train")

            if not post_epoch_status:
                break

    def _epoch_iterator(self, dataloader):
        for scaffold_batch, decorator_batch in dataloader:
            loss = self.model.likelihood(*scaffold_batch, *decorator_batch).mean()

            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_gradient > 0:
                tnnu.clip_grad_norm_(self.model.network.parameters(), self.clip_gradient)

            self.optimizer.step()

            yield loss

    def _initialize_dataloader(self, training_set):
        dataset = md.DecoratorDataset(training_set, vocabulary=self.model.vocabulary)
        return tud.DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                              collate_fn=md.DecoratorDataset.collate_fn, drop_last=True)


class CollectStatsFromModel(Action):
    """Collects stats from an existing RNN model."""

    def __init__(self, model, epoch, training_set, validation_set, writer, sample_size,
                 decoration_type="single", with_weights=False, other_values=None, logger=None):
        """
        Creates an instance of CollectStatsFromModel.
        : param model: A model instance initialized as sampling_mode.
        : param epoch: Epoch number to be sampled(informative purposes).
        : param training_set: Iterator with the training set.
        : param validation_set: Iterator with the validation set.
        : param writer: Writer object(Tensorboard writer).
        : param other_values: Other values to save for the epoch.
        : param sample_size: Number of molecules to sample from the training / validation / sample set.
        : param decoration_type: Kind of decorations (single or all).
        : param with_weights: To calculate or not the weights.
        : return:
        """
        Action.__init__(self, logger)
        self.model = model
        self.epoch = epoch
        self.sample_size = sample_size
        self.training_set = training_set
        self.validation_set = validation_set
        self.writer = writer
        self.other_values = other_values

        self.decoration_type = decoration_type
        self.with_weights = with_weights
        self.sample_size = max(sample_size, 1)

        self.data = {}

        self._calc_nlls_action = CalculateNLLsFromModel(self.model, 128, self.logger)
        self._sample_model_action = SampleModel(self.model, 128, self.logger)

    @torch.no_grad()
    def run(self):
        """
        Collects stats for a specific model object, epoch, validation set, training set and writer object.
        : return: A dictionary with all the data saved for that given epoch.
        """
        self._log("info", "Collecting data for epoch %s", self.epoch)
        self.data = {}

        self._log("debug", "Slicing training and validation sets")
        sliced_training_set = list(random.sample(self.training_set, self.sample_size))
        sliced_validation_set = list(random.sample(self.validation_set, self.sample_size))

        self._log("debug", "Sampling decorations for both sets")
        sampled_training_mols, sampled_training_nlls = self._sample_decorations(next(zip(*sliced_training_set)))
        sampled_validation_mols, sampled_validation_nlls = self._sample_decorations(next(zip(*sliced_validation_set)))

        self._log("debug", "Calculating NLLs for the validation and training sets")
        training_nlls = np.array(list(self._calc_nlls_action.run(sliced_training_set)))
        validation_nlls = np.array(list(self._calc_nlls_action.run(sliced_validation_set)))

        if self.with_weights:
            self._log("debug", "Calculating weight stats")
            self._weight_stats()

        self._log("debug", "Calculating nll stats")
        self._nll_stats(sampled_training_nlls, sampled_validation_nlls, training_nlls, validation_nlls)

        self._log("debug", "Calculating validity stats")
        self._valid_stats(sampled_training_mols, "training")
        self._valid_stats(sampled_validation_mols, "validation")

        self._log("debug", "Drawing some molecules")
        self._draw_mols(sampled_training_mols, "training")
        self._draw_mols(sampled_validation_mols, "validation")

        if self.other_values:
            self._log("debug", "Adding other values")
            for name, val in self.other_values.items():
                self._add_scalar(name, val)

        return self.data

    def _sample_decorations(self, scaffold_list):
        mols = []
        nlls = []
        for scaff, decoration, nll in self._sample_model_action.run(scaffold_list):
            if self.decoration_type == "single":
                mol = usc.join_first_attachment(scaff, decoration)
            elif self.decoration_type == "all":
                mol = usc.join_joined_attachments(scaff, decoration)
            if mol:
                mols.append(mol)
            nlls.append(nll)
        return (mols, np.array(nlls))

    def _valid_stats(self, mols, name):
        self._add_scalar("valid_{}".format(name), 100.0*len(mols)/self.sample_size)

    def _weight_stats(self):
        for name, weights in self.model.network.named_parameters():
            self._add_histogram("weights/{}".format(name), weights.clone().cpu().data.numpy())

    def _nll_stats(self, sampled_training_nlls, sampled_validation_nlls, training_nlls, validation_nlls):
        self._add_histogram("nll_plot/sampled_training", sampled_training_nlls)
        self._add_histogram("nll_plot/sampled_validation", sampled_validation_nlls)
        self._add_histogram("nll_plot/validation", validation_nlls)
        self._add_histogram("nll_plot/training", training_nlls)

        self._add_scalars("nll/avg", {
            "sampled_training": sampled_training_nlls.mean(),
            "sampled_validation": sampled_validation_nlls.mean(),
            "validation": validation_nlls.mean(),
            "training": training_nlls.mean()
        })

        self._add_scalars("nll/var", {
            "sampled_training": sampled_training_nlls.var(),
            "sampled_validation": sampled_validation_nlls.var(),
            "validation": validation_nlls.var(),
            "training": training_nlls.var()
        })

        def bin_dist(dist, bins=1000, dist_range=(0, 100)):
            bins = np.histogram(dist, bins=bins, range=dist_range, density=False)[0]
            bins[bins == 0] = 1
            return bins / bins.sum()

        def jsd(dists, binned=False):  # notice that the dists can or cannot be binned
            # get the min size of each dist
            min_size = min(len(dist) for dist in dists)
            dists = [dist[:min_size] for dist in dists]
            if binned:
                dists = [bin_dist(dist) for dist in dists]
            num_dists = len(dists)
            avg_dist = np.sum(dists, axis=0) / num_dists
            return np.sum([sps.entropy(dist, avg_dist) for dist in dists]) / num_dists

        self._add_scalar("nll_plot/jsd_joined_bins",
                         jsd([sampled_training_nlls, sampled_validation_nlls,
                              training_nlls, validation_nlls], binned=True))

        self._add_scalar("nll_plot/jsd_joined_no_bins",
                         jsd([sampled_training_nlls, sampled_validation_nlls,
                              training_nlls, validation_nlls]))

    def _draw_mols(self, mols, name):
        try:
            utb.add_mols(self.writer, "molecules_{}".format(name), random.sample(
                mols, 16), mols_per_row=4, global_step=self.epoch)
        except ValueError:
            pass

    def _add_scalar(self, key, val):
        self.data[key] = val
        self.writer.add_scalar(key, val, self.epoch)

    def _add_scalars(self, key, dict_vals):
        for k, val in dict_vals.items():
            self.data["{}.{}".format(key, k)] = val
        self.writer.add_scalars(key, dict_vals, self.epoch)

    def _add_histogram(self, key, vals):
        self.data[key] = vals
        self.writer.add_histogram(key, vals, self.epoch)


class SampleModel(Action):

    def __init__(self, model, batch_size, logger=None):
        """
        Creates an instance of SampleModel.
        :params model: A model instance (better in sampling mode).
        :params batch_size: Batch size to use.
        :return:
        """
        Action.__init__(self, logger)
        self.model = model
        self.batch_size = batch_size

    def run(self, scaffold_list):
        """
        Samples the model for the given number of SMILES.
        :params scaffold_list: A list of scaffold SMILES.
        :return: An iterator with each of the batches sampled in (scaffold, decoration, nll) triplets.
        """
        dataset = md.Dataset(scaffold_list, self.model.vocabulary.scaffold_vocabulary,
                             self.model.vocabulary.scaffold_tokenizer)
        dataloader = tud.DataLoader(dataset, batch_size=self.batch_size,
                                    shuffle=False, collate_fn=md.Dataset.collate_fn)
        for batch in dataloader:
            for scaff, dec, nll in self.model.sample_decorations(*batch):
                yield scaff, dec, nll


class CalculateNLLsFromModel(Action):

    def __init__(self, model, batch_size, logger=None):
        """
        Creates an instance of CalculateNLLsFromModel.
        :param model: A model instance.
        :param batch_size: Batch size to use.
        :return:
        """
        Action.__init__(self, logger)
        self.model = model
        self.batch_size = batch_size

    def run(self, scaffold_decoration_list):
        """
        Calculates the NLL for a set of SMILES strings.
        :param scaffold_decoration_list: List with pairs of (scaffold, decoration) SMILES.
        :return: An iterator with each NLLs in the same order as the list.
        """
        dataset = md.DecoratorDataset(scaffold_decoration_list, self.model.vocabulary)
        dataloader = tud.DataLoader(dataset, batch_size=self.batch_size, collate_fn=md.DecoratorDataset.collate_fn,
                                    shuffle=False)
        for scaffold_batch, decorator_batch in dataloader:
            for nll in self.model.likelihood(*scaffold_batch, *decorator_batch).data.cpu().numpy():
                yield nll
