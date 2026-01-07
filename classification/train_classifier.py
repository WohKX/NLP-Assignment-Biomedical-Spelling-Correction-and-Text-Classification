import sys
from pathlib import Path

# --- Runtime bootstrap (fixes 'No module named config') ---
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.runtime import ensure_project_root
ensure_project_root()
# ----------------------------------------------------------

import os
import pickle
import joblib
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn Imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.calibration import CalibratedClassifierCV

# Custom Imports
from utils.preprocess import preprocess_for_classification

# --- CONFIG INTEGRATION ---
from config import (
    LABELED_DATA_PATH,       
    CLASSIF_ARTIFACT_DIR,    
    CLASSIFIER_MODEL_PATH,
    COMPARISON_REPORT_PATH
)

# Set Main Config
DATA_PATH = LABELED_DATA_PATH
RANDOM_STATE = 42

# Optional Visualization
try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None


def clean_text_wrapper(text_series):
    """
    Wrapper function to apply the preprocessing step within a scikit-learn pipeline.
    This ensures the deployed model can accept raw text input.
    """
    # Check if input is a DataFrame or Series, extract values if so
    if isinstance(text_series, pd.DataFrame):
        return text_series.iloc[:, 0].apply(preprocess_for_classification)
    elif isinstance(text_series, pd.Series):
        return text_series.apply(preprocess_for_classification)
    # If list or numpy array
    return [preprocess_for_classification(t) for t in text_series]


def perform_eda(df_train, label_mapping):
    """
    Perform Exploratory Data Analysis strictly on the TRAINING set 
    to prevent data leakage.
    """
    print("\n" + "="*40)
    print(" [EDA] Training Set Analysis ")
    print("="*40)
    
    # 1. Class Distribution
    print("Class Distribution (Train):")
    print(df_train["label"].value_counts())

    # 2. Sequence Length Analysis
    avg_len = df_train.groupby("label")["clean_text"].apply(lambda x: x.str.split().str.len().mean())
    print("\nAverage Token Count per Class (Train):")
    print(avg_len)

    # 3. Top N-Grams per Class
    print("\nTop Unigrams/Bigrams per Class (TF-IDF weighted):")
    for lbl, lid in label_mapping.items():
        subset = df_train[df_train["label_id"] == lid]["clean_text"]
        if subset.empty:
            continue
            
        # Use CountVectorizer for simple freq or Tfidf for importance
        vec = CountVectorizer(ngram_range=(1, 2), max_features=20, stop_words='english')
        try:
            X_counts = vec.fit_transform(subset)
            sum_counts = X_counts.sum(axis=0)
            words_freq = [(word, sum_counts[0, idx]) for idx, word in enumerate(vec.get_feature_names_out())]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
            
            top_terms = ", ".join([w for w, _ in words_freq[:10]])
            print(f"  [{lbl}]: {top_terms}")
            
            # 4. Generate WordCloud (Saved to artifacts)
            if WordCloud is not None:
                text_blob = " ".join(subset.tolist())
                wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text_blob)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                plt.title(f"WordCloud - {lbl} (Train Split)")
                
                img_path = os.path.join(CLASSIF_ARTIFACT_DIR, f"wordcloud_{lbl}.png")
                plt.savefig(img_path, bbox_inches="tight")
                plt.close()
                print(f"    -> Saved WordCloud to {img_path}")
        except ValueError:
            print(f"  [{lbl}]: Not enough data for N-grams.")


def evaluate_model(name, model, X_test_full_df, y_test, label_mapping):
    """
    Evaluate model performance and perform Error Analysis using ORIGINAL text.
    
    Args:
        X_test_full_df: DataFrame containing both 'clean_text' and 'full_text'.
    """
    print("\n" + "=" * 60)
    print(f" Evaluation: {name} ")
    print("=" * 60)
    
    # Predict uses clean_text because the models passed here are trained on clean_text
    X_test_clean = X_test_full_df["clean_text"].values
    y_pred = model.predict(X_test_clean)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    
    print(f"[METRIC] Accuracy: {acc:.4f}")
    print(f"[METRIC] Macro-F1: {f1:.4f}")
    
    target_names = [k for k, v in sorted(label_mapping.items(), key=lambda item: item[1])]
    print("\n[REPORT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("\n[MATRIX] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Error Analysis (The Professor's Touch) ---
    # We use full_text from the dataframe to see the context
    mis_indices = np.where(y_pred != y_test)[0]
    
    if len(mis_indices) > 0:
        print("\n[ANALYSIS] Sample Misclassifications (Original Text):")
        # Invert mapping for display
        inv_map = {v: k for k, v in label_mapping.items()}
        
        # Show top 3 errors
        for i in mis_indices[:3]:
            # Get the original text using iloc on the test dataframe slice
            original_text = X_test_full_df.iloc[i]["full_text"]
            # Clean text for comparison
            processed_text = X_test_full_df.iloc[i]["clean_text"]
            
            true_lbl = inv_map[y_test[i]]
            pred_lbl = inv_map[y_pred[i]]
            
            print(f"  - Example {i}:")
            print(f"    True: [{true_lbl}] | Pred: [{pred_lbl}]")
            print(f"    Orig: {original_text[:150]}...") # First 150 chars
            print(f"    Proc: {processed_text[:100]}...") # Show what the model saw
            print("-" * 30)
            
    return {"Model": name, "Accuracy": acc, "Macro_F1": f1}


def main():
    # Setup directories
    os.makedirs(CLASSIF_ARTIFACT_DIR, exist_ok=True)
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at {DATA_PATH}")
        return

    print("[INFO] Loading labeled dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Basic validation
    required_cols = {"full_text", "label"}
    if not required_cols.issubset(df.columns):
        print(f"[ERROR] Missing columns. Found: {df.columns}")
        return

    # 2. Preprocessing
    # We apply preprocessing globally first for training speed (Optimization),
    # BUT we will wire the preprocessor into the final pipeline for deployment (Safety).
    print("[INFO] Preprocessing texts...")
    df["clean_text"] = df["full_text"].apply(preprocess_for_classification)
    
    # Filter short texts (noise)
    df = df[df["clean_text"].str.split().str.len() > 3].reset_index(drop=True)
    
    label_mapping = {"Prevention": 0, "Treatment": 1, "Epidemiology": 2}
    df["label_id"] = df["label"].map(label_mapping)
    df = df.dropna(subset=["label_id"]) # Drop unmapped labels
    df["label_id"] = df["label_id"].astype(int)

    # 3. Split Data (Before EDA to avoid Leakage)
    # We split the DataFrame to keep metadata (full_text) aligned
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=RANDOM_STATE, 
        stratify=df["label_id"]
    )
    
    print(f"[INFO] Data Split: Train={len(train_df)}, Test={len(test_df)}")

    # 4. EDA (On Training Set Only)
    perform_eda(train_df, label_mapping)

    # Prepare Arrays
    X_train_clean = train_df["clean_text"].values
    y_train = train_df["label_id"].values
    
    # Note: For X_test, we pass the whole DF to evaluation for error analysis
    y_test = test_df["label_id"].values

    # 5. Model Definition
    tfidf = TfidfVectorizer(
    tokenizer=str.split,
    preprocessor=None,
    token_pattern=None,
    lowercase=False,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    )



    # Models
    # A. Logistic Regression (Baseline but strong)
    pipe_lr = Pipeline([
        ("tfidf", tfidf),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))
    ])

    # B. SVM (Often SOTA for text)
    # Using CalibratedClassifierCV to get probabilities from LinearSVC
    svm_base = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, dual="auto")
    pipe_svm = Pipeline([
        ("tfidf", tfidf),
        ("clf", CalibratedClassifierCV(estimator=svm_base)) 
    ])

    # C. Naive Bayes (Fast baseline)
    pipe_nb = Pipeline([
        ("tfidf", tfidf),
        ("clf", MultinomialNB())
    ])

    # 6. Training & Tuning
    print("\n[INFO] Starting Grid Search for Logistic Regression...")
    param_grid_lr = {
        "clf__C": [0.1, 1, 10],
        "tfidf__max_features": [10000, 20000] # Tune vocab size too
    }
    
    grid_lr = GridSearchCV(
        pipe_lr, param_grid=param_grid_lr, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1
    )
    grid_lr.fit(X_train_clean, y_train)
    best_lr = grid_lr.best_estimator_
    print(f"  -> Best LR Params: {grid_lr.best_params_}")

    print("[INFO] Training SVM...")
    pipe_svm.fit(X_train_clean, y_train)
    
    print("[INFO] Training Naive Bayes...")
    pipe_nb.fit(X_train_clean, y_train)

    # 7. Evaluation & Comparison
    results = []
    results.append(evaluate_model("Logistic Regression (Tuned)", best_lr, test_df, y_test, label_mapping))
    results.append(evaluate_model("Linear SVM (Calibrated)", pipe_svm, test_df, y_test, label_mapping))
    results.append(evaluate_model("Multinomial NB", pipe_nb, test_df, y_test, label_mapping))

    # Convert results to DataFrame (Inspired by your other script)
    results_df = pd.DataFrame(results).sort_values(by="Macro_F1", ascending=False)
    print("\n" + "="*40)
    print(" FINAL LEADERBOARD ")
    print("="*40)
    print(results_df)
    
    # Save results CSV
    results_df.to_csv(COMPARISON_REPORT_PATH, index=False)
    print(f"[INFO] Comparison saved to {COMPARISON_REPORT_PATH}")

    # 8. Selection & Deployment Artifact
    best_model_name = results_df.iloc[0]["Model"]
    print(f"\n[INFO] Selecting Best Model: {best_model_name}")
    
    if best_model_name == "Logistic Regression (Tuned)":
        final_model_internal = best_lr
    elif "SVM" in best_model_name:
        final_model_internal = pipe_svm
    else:
        final_model_internal = pipe_nb

    # --- THE DEPLOYMENT GAP FIX ---
    # We construct a NEW pipeline that accepts RAW text.
    # Step 1: Preprocessing (FunctionTransformer)
    # Step 2: Vectorization + Classification (The trained pipeline steps)
    
    # Extract the vectorizer and classifier from the trained pipeline
    trained_tfidf = final_model_internal.named_steps["tfidf"]
    trained_clf = final_model_internal.named_steps["clf"]
    
    deployment_pipeline = Pipeline([
        # The Magic Step: Auto-cleaning raw input
        ("preprocessor", FunctionTransformer(clean_text_wrapper, validate=False)),
        ("tfidf", trained_tfidf),
        ("clf", trained_clf)
    ])
    
    # Sanity Check for Deployment Pipeline
    print("[INFO] Verifying Deployment Pipeline with raw text...")
    sample_raw = ["The patient was treated with Remdesivir and showed improvement."]
    pred_raw = deployment_pipeline.predict(sample_raw)
    print(f"  Input: '{sample_raw[0]}' -> Predicted Class ID: {pred_raw[0]}")

    # Save (joblib payload for robust deployment)
    os.makedirs(CLASSIF_ARTIFACT_DIR, exist_ok=True)
    joblib_path = os.path.join(CLASSIF_ARTIFACT_DIR, "topic_classifier.joblib")

    payload = {
        "model": deployment_pipeline,          # Full RAW->TFIDF->CLF pipeline
        "vectorizer": trained_tfidf,           # Explicit (useful for interpretability / debugging)
        "label_map": label_mapping,
        "metrics": results_df.iloc[0].to_dict(),
        "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__}
    }

    joblib.dump(payload, joblib_path)
    print(f"[SUCCESS] Deployed model saved to: {joblib_path}")
    print("  Note: This model accepts RAW strings. Preprocessing is embedded.")

    # Legacy pickle (optional best-effort for backward compatibility)
    try:
        with open(CLASSIFIER_MODEL_PATH, "wb") as f:
            pickle.dump({
                "model": deployment_pipeline,
                "label_mapping": label_mapping,
                "metrics": results_df.iloc[0].to_dict()
            }, f)
    except Exception:
        pass
if __name__ == "__main__":
    main()