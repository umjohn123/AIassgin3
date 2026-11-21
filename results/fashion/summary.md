# POLCA Net -- Experimental Results

Reconstruction metrics on the test split (NRMSE / PSNR / SSIM) and downstream classification quality (Accuracy / F1, macro-averaged over Perceptron, Ridge, Logistic Regression and Linear SVM).

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
| POLCA Net (full) | fashion | 0.1403 | 17.06 | 0.5595 | 0.6834 | 0.6587 | 0.1671 | 0.8095 |
| w/o L_ort | fashion | 0.1390 | 17.14 | 0.5677 | 0.6756 | 0.6475 | 0.2385 | 0.9762 |
| w/o L_com | fashion | 0.1400 | 17.08 | 0.5590 | 0.6943 | 0.6681 | 0.1490 | 0.6190 |
| w/o L_var | fashion | 0.1404 | 17.05 | 0.5584 | 0.6845 | 0.6706 | 0.1564 | 0.8333 |
