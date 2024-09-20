# MOANA-FL

![MOANA-FL Process](moana_process.pdf)

![MOANA-FL Artifact Correction](correction.pdf)

This repository contains different implementations of federated learning models under the **MOANA-FL** framework. Each folder corresponds to a specific optimization algorithm used in federated learning experiments.

**Directory Structure:**
- *moana-fl-fedavg:* This folder contains experiments and code related to the FedAvg algorithm, a baseline federated learning method where the local models are averaged at each round of training.

- *moana-fl-fedopt:* This folder contains experiments related to the FedOpt algorithm. FedOpt improves on FedAvg by adding adaptive learning rate strategies to speed up convergence and improve model performance.

- *moana-fl-fedprox:* This folder contains experiments for the FedProx algorithm. FedProx adds a proximal term to the loss function, which helps handle system heterogeneity (such as non-IID data or varying computational resources) across the participating devices.


#### Structure:

- **app/config/**: Configuration files for federated clients and the server.
  - `config_fed_client.json`
  - `config_fed_server.json`
  
- **app/custom/**: Contains scripts for the dataset, model, and learning process.
  - `dataset.py`
  - `fed_learner.py`
  - `model_persistor.py`
  - `model.py`
  - `utils.py`

- **meta.json**: Metadata for tracking the experiment.

- **workspaces/**: Contains the workspace setup and client addition scripts.
  - `add_clients.py`
  - `project_original.yml`
  - `project.yml`

- **requirements.txt**: Python dependencies needed to run the FedAvg implementation.

- **start_fl_admin.sh**: Script to start the federated learning admin.
- **start_fl_secure_clients.sh**: Script to launch the secure clients.
- **start_fl_secure_mlflow.sh**: Starts a secure instance of MLFlow for experiment tracking.
- **start_fl_secure_server.sh**: Starts the secure server for federated learning.

- **submit_job.py**: Python script to submit learning jobs.
- **submit_job.sh**: Shell script for submitting jobs.

#### Usage:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the MLflow
sbatch start_fl_secure_mlflow.sh

# Start the server
sbatch start_fl_secure_server.sh

# Start the clients
sbatch start_fl_secure_clients.sh

# Submit a federated learning job
sbatch submit_job.py
