# POLCA Net -- Experimental Results

Reconstruction metrics on the test split (NRMSE / PSNR / SSIM) and downstream classification quality (Accuracy / F1, macro-averaged over Perceptron, Ridge, Logistic Regression and Linear SVM).

## Dataset: mnist

| Method | Latent dims | NRMSE | PSNR | SSIM | Accuracy | F1 | Orth. error | Var. ordering |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PCA | 8 | 0.1946 | 14.22 | 0.4497 | 0.7164 | 0.6998 | 0.0077 | 1.0000 |
| LDA | 8 | 0.2347 | 12.59 | 0.3111 | 0.8423 | 0.8382 | 0.0822 | 1.0000 |
| Autoencoder | 8 | 0.1628 | 15.77 | 0.5847 | 0.7663 | 0.7537 | 0.3168 | 0.2540 |
| POLCA Net | 8 | 0.1697 | 15.41 | 0.5593 | 0.7277 | 0.7087 | 0.1673 | 0.6587 |
| POLCA Net + L_class | 8 | 0.1855 | 14.63 | 0.5218 | 0.9570 | 0.9566 | 0.0952 | 0.5079 |

## Ablation study (POLCA Net, unsupervised)

| Variant | Dataset | NRMSE | PSNR | SSIM | Accuracy | F1 | Orth. error | Var. ordering |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| POLCA Net (full) | mnist | 0.1684 | 15.48 | 0.5653 | 0.7224 | 0.7024 | 0.1893 | 0.2857 |
| w/o L_ort | mnist | 0.1634 | 15.74 | 0.5842 | 0.7759 | 0.7688 | 0.3374 | 0.9762 |
| w/o L_com | mnist | 0.1667 | 15.56 | 0.5714 | 0.7526 | 0.7378 | 0.1018 | -0.4286 |
| w/o L_var | mnist | 0.1680 | 15.50 | 0.5655 | 0.6909 | 0.6752 | 0.1889 | 0.2857 |
