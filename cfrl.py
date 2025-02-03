import tensorflow as tf
import numpy as np
from alibi.explainers import CounterfactualRLTabular
from alibi.models.tensorflow import HeAE
from alibi.explainers.backends.cfrl_tabular import get_statistics
from alibi.models.tensorflow import ADULTEncoder, ADULTDecoder

from sklearn.exceptions import NotFittedError

# Define the function
def generate_rl_counterfactuals(X_train, y_train, X_test, y_test, categorical_ids, numerical_ids, category_map, clf, model_type):

    pp = clf[0]

    # Preprocess training data
    trainset_input = pp.transform(X_train).astype(np.float32)
    trainset_outputs = {"output_1": trainset_input[:, :len(numerical_ids)]}

    for i, cat_id in enumerate(categorical_ids):
        trainset_outputs.update({
            f"output_{i+2}": pp.transform(X_train).astype(np.float32)[:, cat_id]
        })

    trainset = tf.data.Dataset.from_tensor_slices((trainset_input, trainset_outputs))
    trainset = trainset.shuffle(1024).batch(128, drop_remainder=True)

    # Define constants
    EPOCHS = 50
    HIDDEN_DIM = 128
    LATENT_DIM = 15

    # Define output dimensions
    OUTPUT_DIMS = [len(numerical_ids)]
    OUTPUT_DIMS += [len(category_map[cat_id]) for cat_id in categorical_ids]

    # Define the heterogeneous auto-encoder
    heae = HeAE(
        encoder=ADULTEncoder(hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM),
        decoder=ADULTDecoder(hidden_dim=HIDDEN_DIM, output_dims=OUTPUT_DIMS)
    )

    # Define loss functions and weights
    he_loss = [tf.keras.losses.MeanSquaredError()]
    he_loss_weights = [1.]
    for _ in range(len(categorical_ids)):
        he_loss.append(tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        he_loss_weights.append(1./len(categorical_ids))

    # Define metrics
    metrics = {}
    for i in range(len(categorical_ids)):
        metrics.update({f"output_{i+2}": tf.keras.metrics.SparseCategoricalAccuracy()})

    # Compile and train the autoencoder
    heae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=he_loss,
        loss_weights=he_loss_weights,
        metrics=metrics
    )
    heae.fit(trainset, epochs=EPOCHS, verbose=0)

    # Define constants for the explainer
    COEFF_SPARSITY = 0.5
    COEFF_CONSISTENCY = 0.5
    TRAIN_STEPS = 10000
    BATCH_SIZE = 100

    # Define predictor
    if model_type=='sk':
        predictor = lambda x: clf.predict_proba(x)
    elif model_type=='tf':
        predictor = lambda x: clf.predict(x)
    else:
        print('Please specify model')
        return

    # Define preprocessor functions
    def heae_preprocess_transform(X):
        return pp.transform(X).astype(np.float32)

    def heae_preprocess_inv_transform(X):
        #return pp.inverse_transform(X).astype(np.float32)
        try: 
            return np.hstack([
                pp.named_steps['PP'].transformers_[0][1].inverse_transform(X[:, len(numerical_ids):]),
                pp.named_steps['PP'].transformers_[1][1].inverse_transform(X[:, :len(numerical_ids)]).astype(np.float32)
            ])
        except NotFittedError: # when when working only with num data.
            return np.hstack([
                #pp.named_steps['PP'].transformers_[0][1].inverse_transform(X[:, len(numerical_ids):]),
                pp.named_steps['PP'].transformers_[1][1].inverse_transform(X[:, :len(numerical_ids)]).astype(np.float32)
            ])
        # return np.hstack([
        #     pp.named_steps['PP'].transformers_[0][1].inverse_transform(X[:, len(numerical_ids):]),
        #     X[:, :len(numerical_ids)].astype(np.float32)
        # ])

    # Extract the category map
    category_map = {}
    for idx, cat_id in enumerate(categorical_ids):
        encoder = pp.named_steps['PP'].named_transformers_['cat']
        categories = encoder.categories_[idx]
        category_map[cat_id] = categories.tolist()

    feature_names = list(range(X_train.shape[1]))

    # Initialize the explainer
    explainer = CounterfactualRLTabular(
        predictor=predictor,
        encoder=heae.encoder,
        decoder=heae.decoder,
        latent_dim=LATENT_DIM,
        encoder_preprocessor=heae_preprocess_transform,
        decoder_inv_preprocessor=heae_preprocess_inv_transform,
        coeff_sparsity=COEFF_SPARSITY,
        coeff_consistency=COEFF_CONSISTENCY,
        category_map=category_map,
        feature_names=feature_names,
        train_steps=TRAIN_STEPS,
        batch_size=BATCH_SIZE,
        backend="tensorflow"
    )

    explainer.fit(X=X_train)
    explainer.params['stats'] = get_statistics(X_train, heae_preprocess_transform, category_map=category_map)

    y_pred = clf.predict(X_test)
    out = explainer.explain(X_test, 1 - y_pred, C=[])

    return out