# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements adversarial training based on a model and one or multiple attack methods. It incorporates
original adversarial training, ensemble adversarial training, training on all adversarial data and other common setups.
If multiple attacks are specified, they are rotated for each batch. If the specified attacks have as target a different
model, then the attack is transferred. The `ratio` determines how many of the clean samples in each batch are replaced
with their adversarial counterpart.
.. warning:: Both successful and unsuccessful adversarial samples are used for training. In the case of
              unbounded attacks (e.g., DeepFool), this can result in invalid (very noisy) samples being included.
| Paper link: https://arxiv.org/abs/1705.07204
| Please keep in mind the limitations of defences. While adversarial training is widely regarded as a promising,
    principled approach to making classifiers more robust (see https://arxiv.org/abs/1802.00420), very careful
    evaluations are required to assess its effectiveness case by case (see https://arxiv.org/abs/1902.06705).
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, TYPE_CHECKING

import numpy as np
from art.defences.trainer.trainer import Trainer
from tqdm.auto import trange

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class AdversarialTrainerPredefinedAttacks(Trainer):
    """
    Class performing adversarial training based on a model architecture and one or multiple attack methods.
    Incorporates original adversarial training, ensemble adversarial training (https://arxiv.org/abs/1705.07204),
    training on all adversarial data and other common setups. If multiple attacks are specified, they are rotated
    for each batch. If the specified attacks have as target a different model, then the attack is transferred. The
    `ratio` determines how many of the clean samples in each batch are replaced with their adversarial counterpart.
     .. warning:: Both successful and unsuccessful adversarial samples are used for training. In the case of
                  unbounded attacks (e.g., DeepFool), this can result in invalid (very noisy) samples being included.
    | Paper link: https://arxiv.org/abs/1705.07204
    | Please keep in mind the limitations of defences. While adversarial training is widely regarded as a promising,
        principled approach to making classifiers more robust (see https://arxiv.org/abs/1802.00420), very careful
        evaluations are required to assess its effectiveness case by case (see https://arxiv.org/abs/1902.06705).
    """

    def __init__(
        self,
        classifier: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        ratio: float = 0.5,
    ) -> None:
        """
        Create an :class:`.AdversarialTrainer` instance.
        :param classifier: Model to train adversarially.
        :param ratio: The proportion of samples in each batch to be replaced with their adversarial counterparts.
                      Setting this value to 1 allows to train only on adversarial samples.
        """
        super().__init__(classifier=classifier)

        if ratio <= 0 or ratio > 1:
            raise ValueError("The `ratio` of adversarial samples in each batch has to be between 0 and 1.")
        self.ratio = ratio

        self._precomputed_adv_samples: List[np.ndarray] = []
        self.x_augmented: Optional[np.ndarray] = None
        self.y_augmented: Optional[np.ndarray] = None

    def fit(self, x_groups: List[np.ndarray], y_groups: List[np.ndarray], nb_epochs: int = 20,
            **kwargs) -> None:
        """
        Train a model adversarially. See class documentation for more information on the exact procedure.
        :param x_groups: Training set.
        :param y_groups: Labels for the training set.
        :param nb_epochs: Number of epochs to use for trainings.
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
               the target classifier.
        """
        nb_batches = len(x_groups)
        ind = np.arange(x_groups[0].shape[0])

        for _ in trange(nb_epochs, desc="Adversarial training epochs"):
            np.random.shuffle(ind)

            for batch_id in range(nb_batches):
                # Get batch data
                x_batch = x_groups[batch_id][ind].copy()
                y_batch = y_groups[batch_id][ind]

                # Fit batch
                self._classifier.fit(x_batch, y_batch, nb_epochs=1, batch_size=x_batch.shape[0], verbose=0, **kwargs)

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform prediction using the adversarially trained classifier.
        :param x: Input samples.
        :param kwargs: Other parameters to be passed on to the `predict` function of the classifier.
        :return: Predictions for test set.
        """
        return self._classifier.predict(x, **kwargs)