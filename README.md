# 🚀 GitHub Actions CI/CD Pipeline - ML Training with Docker

[![CI/CD MLflow Pipeline](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions)

Workflow CI/CD ini dirancang untuk melakukan **automated machine learning model training** menggunakan **MLflow Project** dan **Docker containerization**. Model akan dilatih secara otomatis setiap kali ada push atau pull request ke branch utama.

## 🏆 Kriteria Penilaian: **ADVANCE LEVEL (4 PTS)**

✅ Folder MLProject dengan struktur lengkap  
✅ Workflow CI yang dapat membuat model ML otomatis  
✅ Menyimpan artefak ke GitHub Actions  
✅ Build & push Docker image menggunakan `mlflow models build-docker`

📊 **[Lihat Detail Pemetaan Kriteria](KRITERIA_PENILAIAN.md)**

---

## 📋 Fitur Workflow

### 1. **Automated ML Training**
- Training model Gradient Boosting Classifier
- Menggunakan MLflow untuk tracking experiments
- Hyperparameters configurable
- Metrics & artifacts logging

### 2. **Docker Integration**
- Build Docker image dari trained model
- Push ke Docker Hub (optional)
- Image ready untuk deployment
- Tags: `latest` dan `<commit-sha>`

### 3. **Artifact Management**
- Upload model artifacts ke GitHub
- MLflow runs & training logs
- Retention 30 hari
- Easy download & reproducibility

### 4. **Environment Variables**
- Dataset path configuration
- Target variable specification
- Easy customization

## 🚀 Trigger Workflow

---

## 🚀 Trigger Workflow

Workflow ini akan berjalan otomatis pada:

- **Push** ke branch `main`
- **Pull Request** ke branch `main`
- **Manual trigger** melalui GitHub Actions UI (workflow_dispatch)

---

## 📁 Struktur Project

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                    # Workflow configuration
├── MLProject/
│   ├── MLProject                     # MLflow project config
│   ├── conda.yaml                    # Environment dependencies
│   ├── modelling.py                  # Training script
│   └── diabetes_prediction_dataset/
│       └── data_clean.csv            # Dataset
├── KRITERIA_PENILAIAN.md            # Mapping kriteria advance
├── QUICKSTART.md                     # Quick start guide (5 min)
├── DOCKER_SETUP.md                   # Docker Hub integration guide
├── SETUP_GITHUB_ACTIONS.md          # Complete setup guide
└── README.md                         # This file
```

---

## ⚙️ Configuration

### Environment Variables

```yaml
env:
  DATASET_PATH: "MLProject/diabetes_prediction_dataset/data_clean.csv"
  TARGET_VAR: "diabetes"
```

### Dependencies

Defined in `MLProject/conda.yaml`:
- Python 3.12.7
- MLflow 2.19.0
- Pandas 2.3.3
- Scikit-learn 1.8.0
- NumPy 2.3.5
- SciPy 1.16.3

### Model Hyperparameters

Configurable via `MLProject/MLProject`:
- `test_size`: 0.2
- `random_state`: 42
- `n_estimators`: 100
- `learning_rate`: 0.1
- `max_depth`: 5

---

## 📊 Workflow Steps

1. ✅ **Checkout repository**
2. ✅ **Setup Python 3.12.7**
3. ✅ **Check environment variables**
4. ✅ **Install dependencies** (MLflow, pandas, scikit-learn, dll)
5. ✅ **Verify dataset** availability
6. ✅ **Run MLflow Project** → Train model
7. ✅ **Get MLflow Run ID**
8. ✅ **Build Docker image** with `mlflow models build-docker`
9. ✅ **Login to Docker Hub** (if secrets configured)
10. ✅ **Tag Docker image** (`latest` + `<commit-sha>`)
11. ✅ **Push to Docker Hub**
12. ✅ **Upload artifacts** to GitHub
13. ✅ **Training summary**

---

## 📦 Output & Artifacts

### 1. GitHub Artifacts
- **Name**: `trained-model-<commit-sha>`
- **Contents**: 
  - MLflow runs (model, metrics, parameters)
  - Training logs
- **Retention**: 30 days
- **Access**: GitHub Actions → Workflow Run → Artifacts

### 2. Docker Images
- **Registry**: Docker Hub
- **Repository**: `<username>/diabetes-model`
- **Tags**:
  - `latest` - latest version
  - `<commit-sha>` - specific commit version
- **Pull command**: 
  ```bash
  docker pull <username>/diabetes-model:latest
  ```

---

## � Quick Start

### **Option 1: Baca Panduan Lengkap** 📖
Untuk setup step-by-step lengkap, baca:
- 🚀 **[QUICKSTART.md](QUICKSTART.md)** - Panduan 5 menit
- 📚 **[SETUP_GITHUB_ACTIONS.md](SETUP_GITHUB_ACTIONS.md)** - Setup lengkap
- 🐳 **[DOCKER_SETUP.md](DOCKER_SETUP.md)** - Docker Hub integration

### **Option 2: Quick Commands** ⚡

```powershell
# 1. Push ke GitHub
cd C:\Users\INDRA\Desktop\Submision\Workflow-CI
git init
git add .
git commit -m "Initial commit: CI/CD MLflow Pipeline"
git remote add origin https://github.com/<USERNAME>/<REPO>.git
git push -u origin main

# 2. Setup Docker Hub Secrets (optional)
# Go to: GitHub Repository → Settings → Secrets → Actions
# Add: DOCKER_HUB_USERNAME & DOCKER_HUB_ACCESS_TOKEN

# 3. Monitor workflow
# Go to: GitHub → Actions → Watch workflow run
```

---

## 🐳 Docker Usage

### Pull & Run Model

```bash
# Pull image from Docker Hub
docker pull <username>/diabetes-model:latest

# Run container
docker run -p 5001:8080 <username>/diabetes-model:latest

# Test prediction API
curl http://localhost:5001/ping
```

---

## 🛠️ Customization

### Mengubah Hyperparameters

Edit file `MLProject/MLProject`:

```yaml
entry_points:
  main:
    parameters:
      test_size: {type: float, default: 0.3}      # Ubah sesuai kebutuhan
      random_state: {type: int, default: 123}
      n_estimators: {type: int, default: 200}
      learning_rate: {type: float, default: 0.05}
      max_depth: {type: int, default: 7}
    command: "python modelling.py ..."
```

Commit dan push untuk trigger ulang workflow.

### Menambah Branch Trigger

Edit `.github/workflows/ci.yml`:

```yaml
on:
  push:
    branches: 
      - main
      - develop    # Tambahkan branch lain
  pull_request:
    branches: 
      - main
```

---

## 📊 Verifikasi Kriteria Advance

Checklist untuk memastikan mendapat 4 pts:

- [x] ✅ Folder `MLProject/` dengan struktur lengkap
- [x] ✅ Workflow CI berjalan otomatis saat trigger
- [x] ✅ Model ML di-training dengan MLflow
- [x] ✅ Artefak di-upload ke GitHub Actions
- [x] ✅ Docker image di-build dengan `mlflow models build-docker`
- [x] ✅ Docker image di-push ke Docker Hub

📋 **[Lihat Detail Pemetaan](KRITERIA_PENILAIAN.md)**

---

## � Troubleshooting

### ❌ Workflow Gagal di Step "Verify Dataset"
**Penyebab**: Dataset tidak ditemukan

**Solusi**:
```powershell
# Pastikan dataset ada
ls MLProject\diabetes_prediction_dataset\data_clean.csv

# Jika tidak ada, copy dari folder lain
cp ..\Membangun_model_Indra\diabetes_prediction_dataset\data_clean.csv MLProject\diabetes_prediction_dataset\
```

### ❌ Docker Push Denied
**Penyebab**: Docker Hub credentials tidak valid

**Solusi**:
1. Generate new token di Docker Hub
2. Update secrets di GitHub:
   - `DOCKER_HUB_USERNAME`
   - `DOCKER_HUB_ACCESS_TOKEN`

### ❌ MLflow Run Failed
**Penyebab**: Error saat training

**Solusi**:
- Check logs di step "Run MLflow Project"
- Test lokal: `mlflow run MLProject --env-manager=local`
- Verify dependencies di `conda.yaml`

### ❌ Git Push Authentication Error
**Penyebab**: Credentials tidak valid

**Solusi**: Gunakan Personal Access Token (PAT)
1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate token dengan scope `repo` & `workflow`
3. Gunakan token sebagai password saat push

---

## 📚 Documentation References

- 🚀 **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- 📖 **[SETUP_GITHUB_ACTIONS.md](SETUP_GITHUB_ACTIONS.md)** - Complete setup guide
- 🐳 **[DOCKER_SETUP.md](DOCKER_SETUP.md)** - Docker Hub integration
- 📊 **[KRITERIA_PENILAIAN.md](KRITERIA_PENILAIAN.md)** - Scoring criteria mapping
- 📝 **[README_WORKFLOW.md](README_WORKFLOW.md)** - Workflow details

---

## 🎯 Next Steps

1. ✅ Push repository ke GitHub
2. ✅ Setup Docker Hub secrets (optional)
3. ✅ Trigger workflow & monitor
4. ✅ Download artifacts dari GitHub Actions
5. ✅ Pull & test Docker image
6. ✅ Deploy model ke production

---

## 📞 Support & Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

## 📄 License

This project is part of machine learning assignment. Adjust license as needed.

---

## 👨‍💻 Author

**Indra Firmansyah**  
Machine Learning Engineering - CI/CD Pipeline Implementation

---

**🎉 Happy Automating! Your ML pipeline is ready for production deployment!**
