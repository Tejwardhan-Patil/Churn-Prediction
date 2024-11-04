# Setup Guide

## Requirements

- Python 3.8+
- R 4.0+
- Docker
- PostgreSQL

### Step 1: Clone the Repository

```bash
git clone https://github.com/repo.git
cd churn_prediction
```

### Step 2: Install Dependencies

For Python:

```bash
pip install -r deployment/api/requirements.txt
```

For R:

```r
install.packages('plumber')
source('deployment/api/packages.R')
```

### Step 3: Run the Application

For Python API:

```bash
python deployment/api/app.py
```

For R API:

```bash
Rscript deployment/api/app.R
```

### Step 4: Docker Setup

```bash
docker-compose up
```
