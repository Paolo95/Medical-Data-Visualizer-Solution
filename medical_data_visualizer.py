import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = np.where(df['weight']/((df['height'] / 100)**2) > 25, 1, 0)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # 7    
    catplot = sns.catplot(
        data=df_cat, 
        kind='bar', 
        x='variable', 
        y='total', 
        hue='value', 
        col='cardio'
    )

    catplot.set_axis_labels("variable", "total")
    catplot.set_titles("cardio = {col_name} ")  
    
    # 8
    fig = catplot.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
                (df['ap_lo'] <= df['ap_hi']) &
                (df['height'] >= df['height'].quantile(0.025)) & 
                (df['height'] <= df['height'].quantile(0.975)) & 
                (df['weight'] >= df['weight'].quantile(0.025)) & 
                (df['weight'] <= df['weight'].quantile(0.975))
                ]
    
    # 12
    corr = df_heat.corr()
    
    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(11, 9))

    # 15

    sns.heatmap(
        corr, 
        mask=mask, 
        annot=True,
        fmt=".1f", 
        cbar_kws={"shrink": 0.6},  # Shrink the color bar
        square=True,      # Keep cells square
    )

    # 16
    fig.savefig('heatmap.png')
    return fig
