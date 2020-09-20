import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import DataLoader


class BaseTextIterDataset(torch.utils.data.IterableDataset):

    """
    Base text dataset from which datasets for particular tasks can be built.

    For example:
    >>> dataset = BaseTextIterDataset(epoch_size=4096, tf_dataset='wikipedia/20190301.en')
    >>> dataloader = DataLoader(dataset, 2)
    >>> for i, batch in enumerate(dataloader):
            print(batch)
        # Philosophy is the study of general and fundamental questions about existence ...
        # Philosophical method (or philosophical methodology) is the study of how to do ...
    """

    def __init__(self, tf_dataset='wikipedia/20190301.en',
                 epoch_size=32_768,
                 split_='train',
                 shuffle_files=True):
        """
        Constructs an instance downloading appropriate data when necessary.

        Parameters
        ----------
        tf_dataset : str
            The relative path to the dataset as listed in the Tensorflow
            Resource catalog:

            https://www.tensorflow.org/datasets/catalog/overview

            In theory, this class should be able to handle any text
            dataset from the catalog. (This has not yet been checked.)
            The default path is 'wikipedia/20190301.en' (the English
            version of Wikipedia captured on March 1, 2019).

        epoch_size : int
            The total number of samples exhausting a single pass of a
            torch dataloader. Note that if the underlying `tf_dataset`
            has not yet been exhausted, the next pass of the dataloader
            will pick up where the previous pass left off. It will not
            begin the dataset from its beginning. Default is 32,768.

        split_ : str
            The catalog lists the splits that are available for each
            dataset. For the default Wikipedia dataset, only a train
            set exists. Default is 'train'.

        shuffle_files : boolean
            Should the dataset files be shuffled or delivered as they
            exist in the dataset directory? Default is True.

        """
        super(BaseTextIterDataset).__init__()
        self.epoch_size = epoch_size
        self.name = tf_dataset

        # Construct a tf.data.Dataset
        ds = tfds.load(tf_dataset, split=split_, shuffle_files=shuffle_files)
        self.ds = ds.batch(self.epoch_size).prefetch(tf.data.experimental.AUTOTUNE)

    def __iter__(self):
        for example in tfds.as_numpy(self.ds.take(1)):
            yield example
