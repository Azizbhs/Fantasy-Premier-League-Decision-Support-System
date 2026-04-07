import os
import pickle
import tensorflow as tf
import autokeras as ak

# Test Huber
try:
    with open('models/pycaret_best_model.pkl', 'rb') as f:
        huber = pickle.load(f)
    print('Huber loaded:', type(huber))
except Exception as e:
    print('Huber error:', e)

# Test LightGBM
try:
    with open('models/best_models.pkl', 'rb') as f:
        best = pickle.load(f)
    print('LightGBM loaded:', type(best[1]))
except Exception as e:
    print('LightGBM error:', e)

# Test AutoKeras
try:
    ak_model = tf.keras.models.load_model(
        'models/autokeras_best_model.keras',
        custom_objects=ak.CUSTOM_OBJECTS
    )
    print('AutoKeras loaded:', type(ak_model))
except Exception as e:
    print('AutoKeras error:', e)