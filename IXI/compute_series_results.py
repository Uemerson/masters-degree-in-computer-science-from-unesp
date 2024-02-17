import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dsc_path = "DSC-validate/"
loss_path = "Loss-train/"

models_dir = [
    "vxm_2_max_mse_1_diffusion_1.csv",
    "vxm_2_bpca_mse_1_diffusion_1.csv",
    "vxm_2_mse_1_diffusion_1.csv",
    "vxm_2_bpca_revert_mse_1_diffusion_1.csv",
    "vxm_2_max_bpca_mse_1_diffusion_1.csv",
    "vxm_2_bpca_max_mse_1_diffusion_1.csv",
]

pooling_names = [
    "Max Pooling",
    "BPCA",
    "Conv. Subamostragem",
    "BPCA Inverso",
    "Max Pooling(3) + BPCA(1)",
    "BPCA(3) + Max Pooling(1)"
]

fig, axs = plt.subplots(2, 3, figsize=(20, 12))

for csv_file, ax, pooling_name in zip(models_dir, axs.flatten(), pooling_names):
    df_dsc = pd.read_csv(f"{dsc_path}{csv_file}")
    df_loss = pd.read_csv(f"{loss_path}{csv_file}")

    ax.plot(df_dsc['Step'], df_dsc['Value'],
            label='Validação',  color='#D95319')
    # ax.plot(df_loss['Step'], df_loss['Value'], label='Loss')

    # Adicione rótulos e título ao gráfico
    ax.set_title(pooling_name)
    ax.set_xlabel('Épocas')
    ax.set_ylabel('Dice Score')
    ax.legend(loc='best')

    print(f"{pooling_name}")
    print(f"min dsc: {df_dsc['Value'].min()}")
    print(f"max dsc: {df_dsc['Value'].max()}")
    print()


fig.subplots_adjust(wspace=0.2, hspace=0.2)
fig.savefig('dice_score_validate.png', bbox_inches='tight')
