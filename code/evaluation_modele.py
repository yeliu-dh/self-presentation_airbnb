import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score


def load_data(path):
    return pd.read_csv(path, sep=',', on_bad_lines='skip', encoding='utf-8')

def stratified_sample(df, col, n_each=10):
    def try_sample(subset):
            try:
                return subset.sample(n=n_each, random_state=1)
            except ValueError:
                return subset  # 如果不足就全取
            
    high = try_sample(df[df[col] > 1.0])
    low = try_sample(df[df[col] < -1.0])
    mid = try_sample(df[(df[col] >= -0.5) & (df[col] <= 0.5)])
    return pd.concat([high, low, mid])

def sampling(path, cols, n_each=10):
    df = load_data(path)
    all_samples = []
    for col in cols:
        sample = stratified_sample(df, col, n_each)
        all_samples.append(sample)
    samples_df=pd.concat(all_samples)
    cols_=['host_about']+cols
    samples_df=samples_df[cols_].dropna(subset='host_about').drop_duplicates(subset="host_about")

    return samples_df.reset_index()

# result = sampling("D:\MASTER_ENC\mini_memoire/res_tactiques2/listings_zsc_tactics8.csv", ["openness", "authenticity",'sociability','self_promotion','exemplification'], n_each=10)
# result.to_csv("D:\Master_ENC\mini_memoire\self-presentation_airbnb\data/sampled_listings.csv",index=False)




