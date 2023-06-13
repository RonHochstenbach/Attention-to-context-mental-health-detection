from load_save_model import load_saved_model_weights, load_params

path = '/Users/ronhochstenbach/Desktop/Thesis/Data/Saved Models/Self-Harm 2023-06-13 07:23:33.542368'

hyperparams, hyperparams_features = load_params(path)

model = load_saved_model_weights(path, hyperparams, hyperparams_features, h5=True)

#model.evaluate??



