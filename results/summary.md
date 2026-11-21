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

## Dataset: mnist

| Method | Latent dims | NRMSE | PSNR | SSIM | Accuracy | F1 | Orth. error | Var. ordering |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PCA | 8 | 0.1946 | 14.22 | 0.4497 | 0.7164 | 0.6998 | 0.0077 | 1.0000 |
| LDA | 8 | 0.2347 | 12.59 | 0.3111 | 0.8423 | 0.8382 | 0.0822 | 1.0000 |
| Autoencoder | 8 | 0.1628 | 15.77 | 0.5847 | 0.7663 | 0.7537 | 0.3168 | 0.2540 |
| POLCA Net | 8 | 0.1697 | 15.41 | 0.5593 | 0.7277 | 0.7087 | 0.1673 | 0.6587 |
| POLCA Net + L_class | 8 | 0.1855 | 14.63 | 0.5218 | 0.9570 | 0.9566 | 0.0952 | 0.5079 |

## Dataset: fashion

| Method | Latent dims | NRMSE | PSNR | SSIM | Accuracy | F1 | Orth. error | Var. ordering |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PCA | 8 | 0.1633 | 15.74 | 0.4680 | 0.6765 | 0.6536 | 0.0078 | 1.0000 |
| LDA | 8 | 0.2434 | 12.27 | 0.3542 | 0.7882 | 0.7721 | 0.0611 | 1.0000 |
| Autoencoder | 8 | 0.1362 | 17.32 | 0.5763 | 0.6902 | 0.6719 | 0.3857 | -0.1270 |
| POLCA Net | 8 | 0.1431 | 16.89 | 0.5502 | 0.6723 | 0.6511 | 0.1369 | 0.7698 |
| POLCA Net + L_class | 8 | 0.1582 | 16.02 | 0.5182 | 0.8567 | 0.8531 | 0.2077 | 0.3254 |

## Ablation study (POLCA Net, unsupervised)

| Variant | Dataset | NRMSE | PSNR | SSIM | Accuracy | F1 | Orth. error | Var. ordering |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| POLCA Net (full) | digits | 0.1247 | 18.08 | 0.9316 | 0.7911 | 0.7847 | 0.1787 | 0.9286 |
| POLCA Net (full) | mnist | 0.1684 | 15.48 | 0.5653 | 0.7224 | 0.7024 | 0.1893 | 0.2857 |
| POLCA Net (full) | fashion | 0.1403 | 17.06 | 0.5595 | 0.6834 | 0.6587 | 0.1671 | 0.8095 |
| w/o L_ort | digits | 0.1191 | 18.48 | 0.9376 | 0.8426 | 0.8425 | 0.3393 | 0.8810 |
| w/o L_ort | mnist | 0.1634 | 15.74 | 0.5842 | 0.7759 | 0.7688 | 0.3374 | 0.9762 |
| w/o L_ort | fashion | 0.1390 | 17.14 | 0.5677 | 0.6756 | 0.6475 | 0.2385 | 0.9762 |
| w/o L_com | digits | 0.1223 | 18.25 | 0.9335 | 0.8113 | 0.8088 | 0.1125 | -0.0238 |
| w/o L_com | mnist | 0.1667 | 15.56 | 0.5714 | 0.7526 | 0.7378 | 0.1018 | -0.4286 |
| w/o L_com | fashion | 0.1400 | 17.08 | 0.5590 | 0.6943 | 0.6681 | 0.1490 | 0.6190 |
| w/o L_var | digits | 0.1251 | 18.06 | 0.9313 | 0.7994 | 0.7908 | 0.1610 | 0.9286 |
| w/o L_var | mnist | 0.1680 | 15.50 | 0.5655 | 0.6909 | 0.6752 | 0.1889 | 0.2857 |
| w/o L_var | fashion | 0.1404 | 17.05 | 0.5584 | 0.6845 | 0.6706 | 0.1564 | 0.8333 |

## Notes

- This file aggregates the three datasets; the per-dataset bundles under results/<dataset>/ carry the same numbers.
