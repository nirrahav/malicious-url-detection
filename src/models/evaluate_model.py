import numpy as np
import pandas as pd
import shap


from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


def generate_classification_report(y_true, y_pred):
    """
    Generates a classification report as a DataFrame.
    """
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    return pd.DataFrame(report).transpose()


def compute_confusion_matrix(y_true, y_pred, class_names):
    """
    Computes a confusion matrix.
    """
    return confusion_matrix(
        y_true,
        y_pred,
        labels=class_names
    )


def get_feature_names(vectorizers, numeric_feature_names=None):
    """
    Combines all feature names into a single ordered list.
    """
    feature_names = []

    if numeric_feature_names is not None:
        feature_names.extend(numeric_feature_names)

    if "char" in vectorizers and vectorizers["char"] is not None:
        feature_names.extend([
            f"char_tfidf__{name}"
            for name in vectorizers["char"].get_feature_names_out()
        ])

    if "token" in vectorizers and vectorizers["token"] is not None:
        feature_names.extend([
            f"token_tfidf__{name}"
            for name in vectorizers["token"].get_feature_names_out()
        ])

    return feature_names


def compute_feature_importance(
    model,
    feature_names,
    top_n=30
):
    """
    Extracts feature importance from a tree-based model.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "Model does not support feature_importances_."
        )

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = (
        importance_df
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return importance_df


def sample_for_shap(
    X,
    sample_size=300,
    random_state=42,
    to_dense=True
):
    """
    Samples rows for SHAP analysis.
    Supports sparse matrices.
    """
    import numpy as np
    from scipy.sparse import issparse

    if issparse(X):
        X = X.tocsr()

    rng = np.random.default_rng(random_state)

    indices = rng.choice(
        X.shape[0],
        size=min(sample_size, X.shape[0]),
        replace=False
    )

    X_sample = X[indices]

    if to_dense and hasattr(X_sample, "toarray"):
        X_sample = X_sample.toarray()

    return X_sample


def compute_shap_values(model, X_sample):
    """
    Computes SHAP values for a tree-based model.
    """

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    return explainer, shap_values