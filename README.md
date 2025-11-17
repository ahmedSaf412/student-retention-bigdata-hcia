# Student Retention Prediction System

## 🔧 **Setup Instructions for Team Members**

### 1. **Clone the Repository**
```bash
git clone https://github.com/ahmedSaf412/student-retention-bigdata-hcia.git
cd student-retention-bigdata-hcia
```

### 2. **Set Up Python Environment**
```bash
# Create your own virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. **Install Docker (ONE TIME ONLY)**
```bash
# If you don't have Docker installed:
sudo apt update
sudo apt install docker.io
sudo usermod -aG docker $USER
```
**👉 LOG OUT AND LOG BACK IN TO YOUR DESKTOP (critical!)**

### 4. **Start the Spark Cluster**
```bash
docker compose up -d
```

### 5. **Verify It's Working**
- Open browser: **http://localhost:8080** → Should show Spark Master UI with 1 worker
- Run: `docker ps` → Should see `spark-master` and `spark-worker-1`

### 6. **Verify Data Access**
```bash
docker exec spark-master ls /opt/spark/data/student_sample.csv
docker exec spark-master ls /opt/spark/models/
```

## 🧩 **Your Role Assignments**

| Role | Task | Starting Point |
|------|------|----------------|
| **Data Engineer** | Improve data pipeline | `spark/preprocess.py` |
| **ML Engineer** | Create prediction script | Use models in `models/` |
| **API Developer** | Build FastAPI endpoint | Create `api/app.py` |
| **DevOps Engineer** | Create Airflow DAG | Create `workflow/dag.py` |
| **BI Analyst** | Create dashboard | Use `figures/` for visuals |

## 📂 **Project Structure**
```
student-retention-bigdata-hcia/
├── Analysis_ML.ipynb       # Complete ML analysis
├── config/                 # Configuration files
├── data/
│   └── student_sample.csv  # ONLY sample data (not full dataset)
├── docker-compose.yml      # Spark cluster configuration
├── Dockerfile              # Custom Spark image
├── figures/                # All visualizations
├── models/                 # Trained models (ready to use)
├── requirements.txt        # Python dependencies
├── spark/
│   └── preprocess.py       # Working preprocessing script
└── README.md               # This file
```

## ⚠️ **Critical Notes**
- ✅ **DO NOT commit** your `venv/` directory
- ✅ **DO NOT commit** full dataset (`data/dataset.csv`)
- ✅ **ALWAYS run** `docker compose up -d` before working
- ✅ **All models are ready** in `models/` — no retraining needed

## 🚀 **Run Your First Test**
```bash
docker exec -it spark-master \
  /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark/spark_scripts/preprocess.py
```

You should see the preprocessing script run successfully.

---
**Project ready for Ministry deployment. Questions? Contact Ahmed.**
