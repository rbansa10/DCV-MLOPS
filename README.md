# DCV-MLOPS
DCV for MLops learning data versioning

Absolutely! Let’s go step by step. I’ll explain **Python DCV (Data-Centric Versioning)** in the context of **MLOps** for **data versioning**, with an example that makes it concrete.

---

## **1. Introduction to Data Versioning in MLOps**

In MLOps, **data is as important as code**. Just like you version control your code using Git, you need to version control your **datasets**, especially when:

* Training models multiple times
* Experimenting with different feature sets
* Ensuring reproducibility
* Collaborating with a team

**Data Version Control (DVC)** is a Python-friendly tool that helps track versions of **large datasets** and **machine learning models**, integrating well with Git.

---

## **2. What is DVC?**

DVC is:

* **Open-source**: Built for ML pipelines
* **Git-friendly**: Works alongside Git for versioning datasets
* **Remote storage compatible**: Supports S3, GCS, Azure, SSH, etc.
* **Pipeline-friendly**: Can track transformations and experiments

**Core idea:** Instead of storing huge data files in Git, DVC stores **pointers to files** and tracks versions in `.dvc` files.

---

## **3. Setting Up DVC in Python Environment**

### **Step 1: Install DVC**

```bash
pip install dvc
```

### **Step 2: Initialize DVC in Project**

```bash
git init          # Initialize git repository if not already done
dvc init          # Initialize DVC in project
```

This creates:

* `.dvc/` folder (DVC config)
* `.dvcignore` file

---

## **4. Example: Versioning a Dataset**

Let’s say we have a dataset `data.csv`.

### **Step 1: Add Data to DVC**

```bash
dvc add data.csv
```

* Creates a `data.csv.dvc` file.
* Replaces large file tracking in Git with a small pointer file.
* Your Git repo now tracks `data.csv.dvc`, not `data.csv` itself.

---

### **Step 2: Commit to Git**

```bash
git add data.csv.dvc .gitignore
git commit -m "Add raw dataset with DVC"
```

**Explanation:**

* `data.csv` is ignored in Git.
* `data.csv.dvc` stores metadata (checksum, size, file path, etc.)
* Ensures reproducibility — you can checkout any dataset version later.

---

### **Step 3: Push Dataset to Remote Storage**

```bash
dvc remote add -d myremote s3://mybucket/ml-datasets
dvc push
```

* Uploads `data.csv` to S3 (or any configured remote storage).
* Only the **pointer file** is in Git.
* Team members can later pull datasets with `dvc pull`.

---

## **5. Tracking Changes in Dataset**

Suppose you clean the dataset and save as `data_clean.csv`.

```bash
dvc add data_clean.csv
git add data_clean.csv.dvc
git commit -m "Add cleaned dataset version"
dvc push
```

* Each version has its own `.dvc` file.
* DVC tracks **history, checksums**, and ensures reproducibility.

---

## **6. Experimenting With ML Models Using Versioned Data**

You can use DVC pipelines to connect **data version** with **training**:

```bash
dvc run -n train_model \
        -d data_clean.csv \
        -d train.py \
        -o model.pkl \
        python train.py
```

* `-n train_model`: pipeline stage name
* `-d`: dependencies (data, scripts)
* `-o`: outputs (trained model)
* Creates a `dvc.yaml` pipeline file.

Now, you can:

* Track which dataset version created which model
* Reproduce results
* Compare experiments using `dvc metrics show`

---

## **7. Reproducing a Previous Version**

```bash
git checkout <commit_hash>
dvc pull
```

* `git checkout` changes the code to previous commit
* `dvc pull` fetches the corresponding dataset version
* Model training can now be fully reproduced

---

## **8. Summary of Key Concepts**

| Concept                 | Explanation                                           |
| ----------------------- | ----------------------------------------------------- |
| `.dvc` files            | Metadata pointers for large data files                |
| `dvc add`               | Tracks dataset, generates `.dvc` pointer              |
| `dvc push` / `dvc pull` | Sync dataset to/from remote storage                   |
| `dvc run` / `dvc.yaml`  | Create reproducible ML pipelines                      |
| Versioning              | Dataset versions can be linked to Git commits         |
| Experiment tracking     | DVC tracks which dataset version produced which model |

---

## **9. Python Integration Example**

You can use DVC with Python scripts like this:

```python
# train.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("data_clean.csv")
X = data.drop("target", axis=1)
y = data["target"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")
print("Model trained and saved!")
```

Then link it with DVC:

```bash
dvc run -n train \
        -d data_clean.csv \
        -d train.py \
        -o model.pkl \
        python train.py
```

* Every time you update `data_clean.csv` and retrain, DVC creates a **new versioned pipeline**.

---

✅ **Key Takeaways:**

1. DVC helps **version large datasets** like Git for code.
2. Maintains **reproducibility** for ML pipelines.
3. Supports **remote storage** for collaborative work.
4. Integrates with Python scripts for **end-to-end ML lifecycle tracking**.
5. Works hand-in-hand with Git for version control.

---

Perfect! Let’s build a **full end-to-end MLOps example with DVC** that demonstrates **data versioning, model versioning, and experiment tracking**. I’ll break it down step by step with Python integration.

---

# **End-to-End MLOps Example with DVC**

### **Scenario**

We have a dataset of house prices (`house_data.csv`) and want to train a **Linear Regression model**. We will:

1. Version raw data
2. Apply data cleaning (create new version)
3. Train model
4. Track experiments and metrics
5. Reproduce previous results

---

## **1. Project Setup**

```bash
mkdir mlops-dvc-example
cd mlops-dvc-example
git init
dvc init
```

Folder structure:

```
mlops-dvc-example/
│
├─ data/
│   ├─ house_data.csv
├─ src/
│   ├─ clean_data.py
│   ├─ train.py
├─ models/
├─ dvc.yaml
├─ .dvc/
├─ .gitignore
```

---

## **2. Version Raw Data**

```bash
dvc add data/house_data.csv
git add data/house_data.csv.dvc .gitignore
git commit -m "Add raw dataset"
```

* `.dvc` file tracks metadata
* Raw data is now versioned

---

## **3. Data Cleaning Script**

**src/clean\_data.py**

```python
import pandas as pd

# Load raw dataset
data = pd.read_csv("../data/house_data.csv")

# Simple cleaning: fill missing values
data = data.fillna(data.mean())

# Save cleaned data
data.to_csv("../data/house_data_clean.csv", index=False)
print("Cleaned data saved!")
```

---

### **4. Track Cleaned Data with DVC**

```bash
dvc add data/house_data_clean.csv
git add data/house_data_clean.csv.dvc
git commit -m "Add cleaned dataset version"
```

* Now we have two dataset versions: raw and cleaned
* DVC ensures reproducibility

---

## **5. Train Model Script**

**src/train.py**

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load cleaned dataset
data = pd.read_csv("../data/house_data_clean.csv")
X = data.drop("price", axis=1)
y = data["price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "../models/linear_model.pkl")

# Save metrics
mse = mean_squared_error(y_test, model.predict(X_test))
with open("../models/metrics.txt", "w") as f:
    f.write(f"MSE: {mse}\n")

print(f"Model trained with MSE={mse}")
```

---

## **6. Create DVC Pipeline**

```bash
dvc run -n clean_data \
        -d src/clean_data.py \
        -d data/house_data.csv \
        -o data/house_data_clean.csv \
        python src/clean_data.py
```

```bash
dvc run -n train_model \
        -d src/train.py \
        -d data/house_data_clean.csv \
        -o models/linear_model.pkl \
        -M models/metrics.txt \
        python src/train.py
```

* `-M` marks metrics file (used for experiment tracking)
* DVC automatically generates `dvc.yaml` showing pipeline stages

---

## **7. Push Data and Models to Remote Storage**

```bash
# Configure remote (example with local path, could be S3/Google Cloud)
dvc remote add -d myremote ../remote_storage
dvc push
```

* Team members can now pull datasets/models using `dvc pull`

---

## **8. Track Experiments**

Suppose you modify `train.py` to try **different test\_size**:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

```bash
dvc repro train_model
```

* Re-runs only affected stages
* Metrics updated in `models/metrics.txt`
* Compare experiments:

```bash
dvc metrics diff
```

* Shows MSE difference between current and previous versions

---

## **9. Reproduce Old Experiment**

```bash
git checkout <previous_commit>
dvc pull
dvc repro
```

* Fetches the exact dataset and regenerates the model
* Ensures **full reproducibility**

---

## **10. Summary of Benefits**

| Feature         | Benefit                                           |
| --------------- | ------------------------------------------------- |
| `dvc add`       | Version raw/processed datasets                    |
| `dvc run`       | Create reproducible ML pipeline                   |
| `dvc push/pull` | Store/fetch datasets and models in remote storage |
| `dvc metrics`   | Track experiment results (like MSE)               |
| `git + DVC`     | Link dataset, code, and model versions            |
| Reproducibility | Reproduce any previous experiment exactly         |

---

✅ **Outcome**

* Full **MLOps workflow with dataset versioning**
* Cleaned data and models tracked
* Experiments recorded with metrics
* Reproducibility guaranteed

---

If you want, I can create a **ready-to-run Python/DVC project template** with **dummy data**, so you can **run it end-to-end on your machine** and practice DVC workflow.

Do you want me to do that?


