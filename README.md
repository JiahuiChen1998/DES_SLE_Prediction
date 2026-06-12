# DES_SLE_Prediction
Machine learning models for predicting the solid-liquid equilibrium (SLE) of LiNO3-based deep eutectic solvent systems.

This work combines models for Redlich-Kister parameters (k0, k1), fusion temperature (Tfus), and fusion enthalpy (Hfus) to screen 10,856 candidate LiNO3-solvent systems and predict their liquidus working windows. The repository includes datasets, training scripts, model files, and predicted results used to support solvent screening and experimental validation.

Project structure
RK Parameters/
Data and training notebooks for predicting Redlich-Kister parameters k0 and k1.

Tfus/
Scripts, data, and training results for fusion temperature prediction.

Hfus/
Scripts, data, and training results for fusion enthalpy prediction.

Fusion properties/
Source code for molecular representation, tokenization, datasets, and model building.

Solvent_Pool.xlsx
Candidate solvent pool used for large-scale screening.

Predicted_Working_Windows.xlsx
Predicted working windows for screened LiNO3-solvent systems.

All predicted binary SLE phase diagrams are listed in https://drive.google.com/drive/folders/1lX7v3B-TlDRQpZfjQ4HRRDSb5139EQXQ?usp=drive_link

If interested in SLE equation please refer our previous work: 1.https://www.sciencedirect.com/science/article/abs/pii/S0009250922006261
                                                              2.https://pubs.acs.org/doi/abs/10.1021/acs.iecr.3c00054

If you find this repo useful please properly cite the corresponding paper. Any related questions, contact alfiechen98@gmail.com. Many thanks! :)
