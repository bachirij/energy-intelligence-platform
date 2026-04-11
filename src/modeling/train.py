"""
Training Orchestrator — Electricity Load Forecasting (h+1)
----------------------------------------------------------

Trains and compares multiple models on the feature-engineered dataset.
Selects the best model on the validation set and evaluates it on the test set.

This file is intentionally thin — it only calls the other modules.
To change behaviour, edit the relevant module:
    - Add/remove models      → models.py
    - Change features/splits → config.py
    - Change metrics         → evaluate.py
    - Change file paths      → storage.py

Temporal splits (updated automatically each year via config.py):
    Train : 2015–2023
    Val   : 2024          ← model selection
    Test  : 2025          ← final honest evaluation (touched once)

Output:
    models/best_model.pkl          ← best model pipeline
    models/training_results.json   ← metrics for all models + metadata

Usage:
    python src/modeling/train.py
    python main.py --steps train
"""

from src.modeling.config   import COUNTRY, YEARS
from src.modeling.models   import build_models
from src.modeling.evaluate import compute_metrics, print_summary
from src.modeling.storage  import load_features, split_data, save_model_and_results


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------

def train_and_compare(X_train, y_train, X_val, y_val) -> tuple:
    """
    Train all models defined in models.py and evaluate on the validation set.

    Returns
    -------
    results : dict
        {model_name: {"val": {"mae": ..., "rmse": ...}}}
    best_name : str
        Name of the model with the lowest validation MAE.
    best_model : fitted Pipeline
        Ready to be evaluated on the test set.
    """
    models   = build_models()
    results  = {}
    best_mae = float("inf")
    best_name, best_model = None, None

    for name, pipeline in models.items():
        print(f"\n[TRAIN] {name} ...")

        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_val)
        metrics = compute_metrics(y_val, y_pred)
        results[name] = {"val": metrics}

        print(f"  VAL  MAE  = {metrics['mae']:>10.2f} MW")
        print(f"  VAL  RMSE = {metrics['rmse']:>10.2f} MW")

        if metrics["mae"] < best_mae:
            best_mae   = metrics["mae"]
            best_name  = name
            best_model = pipeline

    return results, best_name, best_model


# ---------------------------------------------------------------------
# Final test evaluation
# ---------------------------------------------------------------------

def evaluate_on_test(best_model, best_name, X_test, y_test, results) -> dict:
    """
    Evaluate the best model on the held-out test set.
    Test set is touched only once — after model selection is final.

    Adds test metrics to the results dict and saves everything to disk.
    """
    print(f"\n[TEST] Evaluating best model ({best_name}) on test set ...")

    y_pred       = best_model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred)
    results[best_name]["test"] = test_metrics

    print(f"  TEST MAE  = {test_metrics['mae']:>10.2f} MW")
    print(f"  TEST RMSE = {test_metrics['rmse']:>10.2f} MW")

    save_model_and_results(best_model, best_name, results)

    return results


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def run_training(country: str = COUNTRY, years: list = YEARS) -> dict:
    """
    Full training pipeline:
        1. Load features
        2. Split into train / val / test
        3. Train and compare all models on val
        4. Evaluate best model on test
        5. Save model and metrics
    """
    print("=" * 60)
    print("  Training — Electricity Load Forecasting (h+1)")
    print("=" * 60)

    df = load_features(country=country, years=years)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

    results, best_name, best_model = train_and_compare(
        X_train, y_train, X_val, y_val,
    )

    print(f"\n[BEST] Best model on validation: {best_name}")

    results = evaluate_on_test(best_model, best_name, X_test, y_test, results)

    print_summary(results, best_name)

    return results


# ---------------------------------------------------------------------
# Command line execution
# ---------------------------------------------------------------------

if __name__ == "__main__":
    run_training()