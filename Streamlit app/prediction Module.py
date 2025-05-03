# -*- coding: utf-8 -*-
"""
Plant Disease Classification Prediction Module

This module contains functions for predicting plant diseases using deep learning models.
It implements two different protocols for feature fusion and selection from CNN models:
- Protocol 1: Feature selection before fusion
- Protocol 2: Feature fusion before selection

Models used:
- VGG16
- ResNet152
- DenseNet201
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from joblib import load

# Disease class names mapping
CLASS_NAMES = {
    0: "Tomato_Bacterial_spot",
    1: "Tomato_Early_blight",
    2: "Tomato_Late_blight",
    3: "Tomato_Leaf_Mold",
    4: "Tomato_Septoria_leaf_spot",
    5: "Tomato_Spider_mites_Two_spotted_spider_mite",
    6: "Tomato_Target_Spot",
    7: "Tomato_YellowLeaf_Curl_Virus",
    8: "Tomato_mosaic_virus",
    9: "Tomato_healthy"
}

# Constants
MODEL_PATHS = {
    'vgg16': 'models/Vgg16leakyrelu_100e.h5',
    'resnet': 'models/Resnet152_leaky_50e.h5',
    'densenet': 'models/DenseNet_leakyrelu_80e.h5'
}

SCALER_PATHS = {
    'protocol1': {
        'vgg': 'scalers/standerscalerVGG100.joblib',
        'resnet': 'scalers/standerscalerResnet100.joblib',
        'densenet': 'scalers/standerscalerdensenet100.joblib'
    },
    'protocol2': 'scalers/standerscalerfor_allmodels_fusioned98.09_250.joblib'
}

PCA_PATHS = {
    'protocol1': {
        'vgg': 'pca/acp98.28GG100.joblib',
        'resnet': 'pca/acp98.28Resnet100.joblib',
        'densenet': 'pca/acp98.28DenseNet100.joblib'
    },
    'protocol2': 'pca/pcafor_allmodels_fusioned98.09_250.joblib'
}

MODEL_FINAL_PATHS = {
    'protocol1': 'models/svm_for_models_fusioned_after_acp98.28_300total.joblib',
    'protocol2': 'models/svm_for_models_fusioned98.09_250.joblib'
}

def load_model(model_path, layer_name='fc1'):
    """Load a pre-trained model and create a feature extractor"""
    model = tf.keras.models.load_model(model_path, compile=False)
    output = model.get_layer(layer_name).output
    return tf.keras.models.Model(inputs=model.input, outputs=output)

def preprocess_image(image, target_size):
    """Preprocess an image for model input"""
    image = tf.image.resize(image, target_size)
    return image / 255.0

def extract_features(image, model, target_size):
    """Extract features from an image using a model"""
    processed_image = preprocess_image(image, target_size)
    return model.predict(np.expand_dims(processed_image, axis=0))

def predict_disease_protocol1(image_file):
    """
    Predict disease using Protocol 1:
    Feature selection before fusion (PCA on each model's features separately)
    """
    # Load models
    vgg_model = load_model(MODEL_PATHS['vgg16'])
    resnet_model = load_model(MODEL_PATHS['resnet'])
    densenet_model = load_model(MODEL_PATHS['densenet'])

    # Load scalers and PCA models
    scalers = {k: load(v) for k, v in SCALER_PATHS['protocol1'].items()}
    pca_models = {k: load(v) for k, v in PCA_PATHS['protocol1'].items()}

    # Load final SVM model
    svm_model = load(MODEL_FINAL_PATHS['protocol1'])

    # Process uploaded image
    image = tf.io.decode_image(image_file.read(), channels=3)
    st.image(image.numpy(), caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        # Extract features from each model
        vgg_features = extract_features(image, vgg_model, (224, 224))
        resnet_features = extract_features(image, resnet_model, (200, 200))
        densenet_features = extract_features(image, densenet_model, (224, 224))

        # Scale and transform features
        vgg_scaled = scalers['vgg'].transform(vgg_features)
        vgg_transformed = pca_models['vgg'].transform(vgg_scaled)

        resnet_scaled = scalers['resnet'].transform(resnet_features)
        resnet_transformed = pca_models['resnet'].transform(resnet_scaled)

        densenet_scaled = scalers['densenet'].transform(densenet_features)
        densenet_transformed = pca_models['densenet'].transform(densenet_scaled)

        # Concatenate transformed features
        combined_features = np.concatenate(
            (vgg_transformed, resnet_transformed, densenet_transformed),
            axis=1
        )

        # Make prediction
        prediction = svm_model.predict(combined_features)
        prediction = int(prediction[0])
        
        if prediction in CLASS_NAMES:
            st.success(f'Predicted disease: {CLASS_NAMES[prediction]}')
        else:
            st.error('Predicted class not found in dictionary.')

def predict_disease_protocol2(image_file):
    """
    Predict disease using Protocol 2:
    Feature fusion before selection (PCA on concatenated features)
    """
    # Load models
    vgg_model = load_model(MODEL_PATHS['vgg16'])
    resnet_model = load_model(MODEL_PATHS['resnet'])
    densenet_model = load_model(MODEL_PATHS['densenet'])

    # Load scaler and PCA model
    scaler = load(SCALER_PATHS['protocol2'])
    pca_model = load(PCA_PATHS['protocol2'])

    # Load final SVM model
    svm_model = load(MODEL_FINAL_PATHS['protocol2'])

    # Process uploaded image
    image = tf.io.decode_image(image_file.read(), channels=3)
    st.image(image.numpy(), caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        # Extract features from each model
        vgg_features = extract_features(image, vgg_model, (224, 224))
        resnet_features = extract_features(image, resnet_model, (200, 200))
        densenet_features = extract_features(image, densenet_model, (224, 224))

        # Concatenate features before transformation
        combined_features = np.concatenate(
            (densenet_features, resnet_features, vgg_features),
            axis=1
        )

        # Scale and transform features
        scaled_features = scaler.transform(combined_features)
        transformed_features = pca_model.transform(scaled_features)

        # Make prediction
        prediction = svm_model.predict(transformed_features)
        prediction = int(prediction[0])
        
        if prediction in CLASS_NAMES:
            st.success(f'Predicted disease: {CLASS_NAMES[prediction]}')
        else:
            st.error('Predicted class not found in dictionary.')

# For backward compatibility
predict_disease2 = predict_disease_protocol1
predict_disease3 = predict_disease_protocol2