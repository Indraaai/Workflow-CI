# GitHub Actions CI Workflow

Workflow CI ini dirancang untuk melakukan testing, validasi, dan packaging machine learning pipeline secara otomatis.

## 📋 Fitur Workflow

### 1. **Test Job**

- Menjalankan di Python 3.12
- Instalasi dependencies otomatis
- Linting dengan flake8
- Validasi struktur project
- Pengecekan ketersediaan dataset
- Running tests (jika tersedia)
- Coverage reporting

### 2. **Validate MLflow Job**

- Validasi konfigurasi MLflow project
- Pengecekan conda.yaml dan python_env.yaml
- Verifikasi MLProject setup

### 3. **Build Job**

- Package artifacts
- Upload build artifacts
- Retention 30 hari

### 4. **Security Scan Job**

- Bandit security scanning
- Safety check untuk dependencies
- Upload security reports

### 5. **Notification Job**

- Reporting status dari semua jobs

## 🚀 Trigger Workflow

Workflow ini akan berjalan otomatis pada:

- **Push** ke branch `main` atau `develop`
- **Pull Request** ke branch `main` atau `develop`
- **Manual trigger** melalui GitHub Actions UI (workflow_dispatch)

## 📁 Struktur yang Diperlukan

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml
├── MLProject/
│   ├── conda.yaml
│   ├── python_env.yaml
│   ├── modelling.py
│   └── diabetes_prediction_dataset/
│       └── data_clean.csv
└── README.md
```

## ⚙️ Konfigurasi

### Dependencies

Dependencies didefinisikan dalam `Workflow-CI/MLProject/conda.yaml`:

- mlflow==2.19.0
- cloudpickle==3.1.2
- numpy==2.3.5
- pandas==2.3.3
- psutil==7.1.3
- pyarrow==18.1.0
- scikit-learn==1.8.0
- scipy==1.16.3

### Python Version

- Python 3.12.7

## 📊 Output & Artifacts

### Build Artifacts

- Name: `ml-pipeline-artifacts`
- Contains: MLProject files
- Retention: 30 days

### Security Reports

- Name: `security-reports`
- Contains: Bandit scan results
- Retention: 30 days

## 🔍 Status Badges

Tambahkan badge berikut ke README.md utama project:

```markdown
![CI Status](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/CI%20-%20Machine%20Learning%20Pipeline/badge.svg)
```

Ganti `YOUR_USERNAME` dan `YOUR_REPO` dengan username dan nama repository GitHub Anda.

## 📝 Cara Menggunakan

1. **Push ke GitHub**:

   ```bash
   git add .
   git commit -m "Add CI workflow"
   git push origin main
   ```

2. **Monitor di GitHub Actions**:

   - Buka repository di GitHub
   - Klik tab "Actions"
   - Lihat workflow runs

3. **Manual Trigger**:
   - Buka tab "Actions"
   - Pilih "CI - Machine Learning Pipeline"
   - Klik "Run workflow"
   - Pilih branch dan klik "Run workflow"

## 🛠️ Customization

### Menambah Branch Trigger

Edit `ci.yml`:

```yaml
on:
  push:
    branches: [main, develop, staging] # Tambahkan branch
  pull_request:
    branches: [main, develop, staging]
```

### Menambah Python Version

Edit `ci.yml`:

```yaml
strategy:
  matrix:
    python-version: ["3.10", "3.11", "3.12"] # Tambah versi
```

### Menambah Tests

Buat folder `tests/` dan tambahkan test files:

```
tests/
├── __init__.py
├── test_modelling.py
└── test_preprocessing.py
```

## 🔧 Troubleshooting

### Workflow Gagal pada Lint Stage

- Perbaiki syntax errors yang dilaporkan
- Sesuaikan line length jika perlu

### Dataset Tidak Ditemukan

- Pastikan file `data_clean.csv` ada di lokasi yang benar
- Upload dataset ke repository (jika size < 100MB)
- Atau gunakan Git LFS untuk file besar

### MLflow Validation Gagal

- Check `conda.yaml` format
- Pastikan semua dependencies tersedia
- Verifikasi MLProject structure

## 📚 Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [pytest Documentation](https://docs.pytest.org/)

## 📄 License

Sesuaikan dengan license project Anda.
