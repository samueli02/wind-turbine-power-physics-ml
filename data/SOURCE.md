# Input data provenance

- Dataset: [Wind Turbine Scada Dataset, Kaggle](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset), uploaded by Berker Isen.
- File used: `T1.csv` (50,530 rows; 2018 turbine measurements at roughly ten-minute intervals).
- A public mirror of this same CSV was used for the local submission package: [Dominodatalab reference project](https://github.com/dominodatalab/reference-project-wind-turbine/blob/release-1.0.0/data/T1.csv).
- SHA-256 of the packaged `T1.csv`: `F92C33C1CC199756B759FEC251EEDDC506F35D6A9E47EDE034FC3E339A7C46D5`.
- The source CSV labels `Theoretical_Power_Curve (KWh)`. This project interprets the values as instantaneous **kW** because the measurements are instantaneous kW, the curve caps at 3,600, and a published study of this SCADA dataset describes theoretical power in kW. This is an interpretation, not a correction to the source file.
- The file has no turbine-operation-status, air-density, rotor-area, or power-coefficient columns. Those quantities are not invented here.

The full CSV is included in the local course submission ZIP. The public GitHub repository keeps this provenance note and the download link rather than copying the external dataset.
