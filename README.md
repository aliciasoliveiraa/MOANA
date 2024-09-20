# MOANA-FL

**MOANA-FL** is a framework designed to improve the accuracy and reliability of Deep Learning models by addressing the presence of artifacts in medical imaging data. The framework specifically targets brain MRI scans, where data can suffer from various types of corruptions, including motion, aliasing, magnetic susceptibility, and noise, that affect model performance.

While Federated Learning (FL) is utilized within **MOANA-FL**, it is applied only during the artifact correction phase. By distributing the artifact correction process across multiple clients, **MOANA-FL** ensures privacy-preserving correction of the corrupted MRI scans, restoring them closer to their original state. This allows for more accurate downstream tasks, such as diagnosis or further analysis.

The figures below demonstrate the full **MOANA-FL** process and highlight the effectiveness of the artifact correction model.
The figure above illustrates the complete process of the **MOANA-FL** framework. It starts with data preprocessing, where raw data is prepared for training. This is followed by the simulation model of artifacts, and the correction model is applied to mitigate the impact of the artifacts introduced during training.

![MOANA-FL Process](https://github.com/aliciasoliveiraa/MOANA-FL/blob/main/moana_process.png)

The second figure demonstrates the effectiveness of the artifact correction model. It showcases three images of each contrast (T1w, T1CE, T2w, and FLAIR):
1. **Artifact-free Image** – The original, clean input without any distortions.
2. **Artifact-corrupted Image** – An image that has been affected by artifacts during the FL process, which can compromise model accuracy.
3. **Corrected Image** – The output after the artifact correction model has been applied, showing how the corruption is mitigated, bringing the image closer to the original artifact-free version.

![MOANA-FL Artifact Correction](https://github.com/aliciasoliveiraa/MOANA-FL/blob/main/correction.png)

This repository contains different implementations of FL models under the **MOANA-FL** framework. Each folder corresponds to a specific optimization algorithm used in FL experiments.

**Directory Structure:**
- *moana-fl-fedavg:* This folder contains experiments and code related to the FedAvg algorithm, a baseline FL method where the local models are averaged at each round of training.

- *moana-fl-fedopt:* This folder contains experiments related to the FedOpt algorithm. FedOpt improves on FedAvg by adding adaptive learning rate strategies to speed up convergence and improve model performance.

- *moana-fl-fedprox:* This folder contains experiments for the FedProx algorithm. FedProx adds a proximal term to the loss function, which helps handle system heterogeneity across the participating devices.


#### Structure:

- **app/config/**: Configuration files for FL-clients and the server.
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

- **start_fl_admin.sh**: Script to start the FL admin.
- **start_fl_secure_clients.sh**: Script to launch the secure clients.
- **start_fl_secure_mlflow.sh**: Starts a secure instance of MLFlow for experiment tracking.
- **start_fl_secure_server.sh**: Starts the secure server for FL.

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

# Submit a FL job
sbatch submit_job.py
