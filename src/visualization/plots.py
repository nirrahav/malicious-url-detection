import matplotlib.pyplot as plt
import shap


from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(
    cm,
    class_names,
    title="Confusion Matrix"
):
    """
    Plots a confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(ax=ax, xticks_rotation=45)

    plt.title(title)
    plt.show()


def plot_feature_importance(
    importance_df,
    title="Feature Importance"
):
    """
    Plots feature importance.
    """
    plt.figure(figsize=(10, 8))

    plt.barh(
        importance_df["feature"][::-1],
        importance_df["importance"][::-1]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(title)

    plt.show()


def plot_shap_summary(
    shap_values,
    X_sample,
    feature_names,
    model,
    class_name,
    max_display=20
):
    """
    Plots SHAP summary plot for a specific class in multiclass classification.
    Supports both old and new SHAP output formats.
    """
    import numpy as np
    import shap

    class_index = list(model.classes_).index(class_name)

    if isinstance(shap_values, list):
        class_shap_values = shap_values[class_index]
    else:
        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 3:
            class_shap_values = shap_values[:, :, class_index]
        else:
            class_shap_values = shap_values

    print("X_sample shape:", X_sample.shape)
    print("SHAP class shape:", class_shap_values.shape)

    shap.summary_plot(
        class_shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=max_display
    )