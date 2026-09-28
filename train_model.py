import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("construction_project_delay_final.csv")

print("Dataset loaded:", df.shape)


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

features = [
    "Project_Category",
    "Project_Size",
    "Budget",
    "Workforce",
    "Material_Delivery",
    "Weather",
    "Contractor",
    "Milestone",
    "Planned_Progress"
]

target = "Delay_Status"

X = df[features]
y = df[target]


# ==========================================
# 3. CATEGORICAL / NUMERICAL FEATURES
# ==========================================

categorical_features = [
    "Project_Category",
    "Weather",
    "Contractor",
    "Milestone"
]

numerical_features = [
    "Project_Size",
    "Budget",
    "Workforce",
    "Material_Delivery",
    "Planned_Progress"
]


# ==========================================
# 4. PREPROCESSOR
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ==========================================
# 5. RANDOM FOREST
# ==========================================

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1
)


# ==========================================
# 6. PIPELINE
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf_model)
    ]
)


# ==========================================
# 7. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 8. TRAIN
# ==========================================

model.fit(X_train, y_train)

print("✅ Model trained successfully!")


# ==========================================
# 9. TEST
# ==========================================

predictions = model.predict(X_test)

print("\nSample predictions:")
print(predictions[:10])


# ==========================================
# 10. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "construction_delay_random_forest.pkl"
)

print("\n✅ New model saved successfully!")
print("File: construction_delay_random_forest.pkl")