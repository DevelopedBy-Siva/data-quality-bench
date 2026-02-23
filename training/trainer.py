import random
import numpy as np
import torch
from models.logreg import LogRegModel
from models.distilbert import DistilBertModel
from models.deberta import DebertaModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model(model_name):
    """Returns a fresh untrained model. model_name is 'logreg', 'distilbert', or 'deberta'."""
    if model_name == "logreg":
        return LogRegModel()
    elif model_name == "distilbert":
        return DistilBertModel()
    elif model_name == "deberta":
        return DebertaModel()
    else:
        raise ValueError(
            f"unknown model: {model_name}. choose 'logreg', 'distilbert', or 'deberta'."
        )


def train(model_name, train_texts, train_labels, seed):
    """Sets the seed, builds a fresh model, trains it, returns it."""
    set_seed(seed)
    model = get_model(model_name)
    model.fit(train_texts, train_labels)
    return model
