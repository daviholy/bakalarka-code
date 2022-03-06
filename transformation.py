import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from datasetCreator import DatasetCreator
from scipy.stats import kendalltau

def transform_data(data, NaN=False):
    normal = data[["time","en_total","pe_total","be_total","pressure","s_xx","s_xy","s_xz","s_yx","s_yy","s_yz","s_zx","s_zy","s_zz","temperature","bndlen_av","bndlen_max","bndlen_min","len"]]
    std = MinMaxScaler().fit_transform(normal.to_numpy())
    normal = pd.DataFrame(std, columns=normal.keys())
    normal["f_a"] = data["f_a"]

    """en_total_gm = GaussianMixture(n_components=7 )
    en_total_gm.fit(data["be_total"].values.reshape(-1,1))
    en_target = en_total_gm.predict(data["pe_total"].values.reshape(-1,1))

    for i in range(len(np.unique(en_target))):
        data[f"en_total_{i}"] = np.where(en_target == i, data["en_total"],np.NaN if NaN else 0)

    pe_total_gm = GaussianMixture(n_components=7 )
    pe_total_gm.fit(data["pe_total"].values.reshape(-1,1))
    pe_target = pe_total_gm.predict(data["pe_total"].values.reshape(-1,1))

    for i in range(len(np.unique(pe_target))):
        data[f"pe_total_{i}"] = np.where(pe_target == i, data["pe_total"], np.NaN if NaN else 0)

    be_total_gm = GaussianMixture(n_components=7 )
    be_total_gm.fit(data["be_total"].values.reshape(-1,1))
    be_target = be_total_gm.predict(data["pe_total"].values.reshape(-1,1))

    for i in range(len(np.unique(be_target))):
        data[f"be_total_{i}"] = np.where(be_target == i, data["be_total"], np.NaN if NaN else 0)

    bndlen_av_gm = GaussianMixture(n_components=7 )
    bndlen_av_gm.fit(data["bndlen_av"].values.reshape(-1,1))
    bndlen_av_target = bndlen_av_gm.predict(data["bndlen_av"].values.reshape(-1,1))

    for i in range(len(np.unique(bndlen_av_target))):
        data[f"bndlen_av_{i}"] = np.where(bndlen_av_target == i, data["bndlen_av"], np.NaN if NaN else 0) """
    return normal

if __name__ == "__main__":
    data, labels = DatasetCreator.read_dataset("./dataset")
    data = transform_data(data, NaN=True)

    # for key in data.columns:
    #    sns.histplot(data=data, x=key, kde=True )

    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
    #cols = ["time","en_total","pe_total","be_total","pressure","s_xx","s_xy","s_xz","s_yx","s_yy","s_yz","s_zx","s_zy","s_zz","temperature","bndlen_av","bndlen_max","bndlen_min","f_a"]
    ax.set_title("Correlation Matrix", fontsize=16)
    sns.heatmap(data.corr(method="kendall"), vmin=-1, vmax=1, cmap='coolwarm', annot=True)
    plt.show()





    # f, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    # sns.kdeplot(data=data.pe_total, ax=ax[0])
    # sns.kdeplot(data=data.pe_total[pe_target == 0], ax=ax[1])
    # sns.kdeplot(data=data.pe_total[pe_target == 1], ax=ax[1])
    # sns.kdeplot(data=data.pe_total[pe_target == 2], ax=ax[1])
    # sns.kdeplot(data=data.pe_total[pe_target == 3], ax=ax[1])
    # sns.kdeplot(data=data.pe_total[pe_target == 4], ax=ax[1])
    # sns.kdeplot(data=data.pe_total[pe_target == 5], ax=ax[1])
    # sns.kdeplot(data=data.pe_total[pe_target == 6], ax=ax[1])
    # plt.show()

    # f, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    # sns.kdeplot(data=data.be_total, ax=ax[0])
    # sns.kdeplot(data=data.be_total[be_target == 0], ax=ax[1])
    # sns.kdeplot(data=data.be_total[be_target == 1], ax=ax[1])
    # sns.kdeplot(data=data.be_total[be_target == 2], ax=ax[1])
    # sns.kdeplot(data=data.be_total[be_target == 3], ax=ax[1])
    # sns.kdeplot(data=data.be_total[be_target == 4], ax=ax[1])
    # sns.kdeplot(data=data.be_total[be_target == 5], ax=ax[1])
    # sns.kdeplot(data=data.be_total[be_target == 6], ax=ax[1])
    # plt.show()

    # f, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    # sns.kdeplot(data=data.en_total, ax=ax[0])
    # sns.kdeplot(data=data.en_total[en_target == 0], ax=ax[1])
    # sns.kdeplot(data=data.en_total[en_target == 1], ax=ax[1])
    # sns.kdeplot(data=data.en_total[en_target == 2], ax=ax[1])
    # sns.kdeplot(data=data.en_total[en_target == 3], ax=ax[1])
    # sns.kdeplot(data=data.en_total[en_target == 4], ax=ax[1])
    # sns.kdeplot(data=data.en_total[en_target == 5], ax=ax[1])
    # sns.kdeplot(data=data.en_total[en_target == 6], ax=ax[1])
    # plt.show()

    # f, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    # sns.kdeplot(data=data.bndlen_av, ax=ax[0])
    # sns.kdeplot(data=data.bndlen_av[bndlen_av_target == 0], ax=ax[1])
    # sns.kdeplot(data=data.bndlen_av[bndlen_av_target == 1], ax=ax[1])
    # sns.kdeplot(data=data.bndlen_av[bndlen_av_target == 2], ax=ax[1])
    # sns.kdeplot(data=data.bndlen_av[bndlen_av_target == 3], ax=ax[1])
    # sns.kdeplot(data=data.bndlen_av[bndlen_av_target == 4], ax=ax[1])
    # sns.kdeplot(data=data.bndlen_av[bndlen_av_target == 5], ax=ax[1])
    # sns.kdeplot(data=data.bndlen_av[bndlen_av_target == 6], ax=ax[1])
    # plt.show()

    # cols2 = ['pe_total_0','pe_total_1','pe_total_2','pe_total_3','pe_total_4','pe_total_5','pe_total_6','labels']
    # cols3 = ['be_total_0','be_total_1','be_total_2','be_total_3','be_total_4','be_total_5','be_total_6','labels']
    # cols4 = ['en_total_0','en_total_1','en_total_2','en_total_3','en_total_4','en_total_5','en_total_6','labels']
    # cols5 = ['bndlen_av_0','bndlen_av_1','bndlen_av_2','bndlen_av_3','bndlen_av_4','bndlen_av_5','bndlen_av_6','labels']
    # for col in [cols2, cols3, cols4, cols5]:
    #     f, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
    #     ax.set_title("Correlation Matrix", fontsize=16)
    #     sns.heatmap(data[col].corr(), vmin=-1, vmax=1, cmap='coolwarm', annot=True)
    #     plt.show()


    for col in data.columns:
        tau , p_value = kendalltau(data[col],labels, method= 'asymptotic', nan_policy='omit')
        print(f"{col}:")
        print(f"tau: {tau}      p value:{p_value}")
