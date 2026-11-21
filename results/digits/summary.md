# POLCA Net -- Experimental Results

Reconstruction metrics on the test split (NRMSE / PSNR / SSIM) and downstream classification quality (Accuracy / F1, macro-averaged over Perceptron, Ridge, Logistic Regression and Linear SVM).

## Dataset: digits

| Method | Latent dims | NRMSE | PSNR | SSIM | Accuracy | F1 | Orth. error | Var. ordering |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PCA | 8 | 0.1554 | 16.17 | 0.8908 | 0.8737 | 0.8660 | 0.0446 | 0.9921 |
| LDA | 8 | 0.3277 | 9.70 | 0.4874 | 0.9183 | 0.9143 | 0.0886 | 1.0000 |
| Autoencoder | 8 | 0.1183 | 18.54 | 0.9376 | 0.8661 | 0.8547 | 0.3679 | 0.1984 |
| POLCA Net | 8 | 0.1284 | 17.83 | 0.9267 | 0.8076 | 0.7956 | 0.0910 | 0.7778 |
| POLCA Net + L_class | 8 | 0.1312 | 17.64 | 0.9239 | 0.9485 | 0.9470 | 0.1897 | 0.5873 |

## Ablation study (POLCA Net, unsupervised)

| Variant | Dataset | NRMSE | PSNR | SSIM | Accuracy | F1 | Orth. error | Var. ordering |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| POLCA Net (full) | digits | 0.1247 | 18.08 | 0.9316 | 0.7911 | 0.7847 | 0.1787 | 0.9286 |
| w/o L_ort | digits | 0.1191 | 18.48 | 0.9376 | 0.8426 | 0.8425 | 0.3393 | 0.8810 |
| w/o L_com | digits | 0.1223 | 18.25 | 0.9335 | 0.8113 | 0.8088 | 0.1125 | -0.0238 |
| w/o L_var | digits | 0.1251 | 18.06 | 0.9313 | 0.7994 | 0.7908 | 0.1610 | 0.9286 |
