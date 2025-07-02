# PEP-client

A Python client for launching and managing experiments of the <strong>P</strong>UF-<strong>E</strong>valuation <strong>P</strong>latform. This client allows users to execute remote tests, collect results, compute quality metrics, and manage device power settings over a network.

---

## 📦 Repository Structure

```
PEP-client/
│
├── app.py                # Script for interactive use
├── main.py               # Script for automated use
├── core/
│   ├── AccessData.py     # Contains default access configuration (username, password, etc.)
│   ├── Utility.py        # Utility functions (e.g., logo display, server setup)
│   ├── cert.pem          # SSL certificate (likely used for secure communication)
│   ├── Handlers/         # Modular logic for each operation
│   │   ├── HandlerAccess.py
│   │   ├── HandlerPower.py
│   │   ├── HandlerResults.py
│   │   ├── HandlerTests.py
│   └── Metrics/          # Placeholder for metric computation logic (if implemented)
│
├── tests/                # Experiment data or outputs from test campaigns
│   ├── test0/
│   ├── test1/
│   └── ...
└── .git/                 # Git repository metadata
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <REPO-URL>
cd PEP-client
```

### 2. Install requirements

The project uses external libraries, specified in the `requirements.txt` file of the main repository https://github.com/daniref/PUF-Evaluation-Platform:

```bash
cd ..
pip install -r requirements.txt
```

---

## 🧪 Usage

Use `main.py` for automated execution. The following example powers up devices using the private network:

```bash
python main.py 3 --serverAddress priv --username myuser --password mypass
```

### Available operations (`op` argument):

| Value | Description                    | Extra Arguments                   |
|-------|--------------------------------|-----------------------------------|
| 0     | Launch experiments             | `--numDevices`, `--idTest`        |
| 1     | Download experiment results    | `--idTest`                        |
| 2     | Compute quality metrics        | `--idTest`                        |
| 3     | Power up all devices           |                                   |
| 4     | Power down all devices         |                                   |
| 5     | Power up fans                  |                                   |
| 6     | Power down fans                |                                   |
| 7     | Register a new user            |                                   |

Or use `app.py`, following the provided instructions during its usage, for an interactive experience.

---

## 🔐 Authentication

All operations require a valid `--username` and `--password`. These credentials are used to authenticate with the PUF server.

---

## 🧪 Test Data

The `tests/` folder contains subdirectories (e.g., `test0`, `test1`, ...) which appear to store output data from different experiment campaigns.

---

## 📬 Contact

For questions or access to the corresponding PUF Evaluation Platform, contact the system administrator or project maintainer.