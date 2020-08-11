import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
bmi = df['weight'] / np.power(df['height'] * 0.01, 2.0)
df['overweight'] = [1 if i >= 25 else 0 for i in bmi]

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1,
# make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = [1 if i > 1 else 0 for i in df['cholesterol']]
df['gluc'] = [1 if i > 1 else 0 for i in df['gluc']]


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke',
    # 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars='cardio',
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename
    # one of the columns for the catplot to work correctly.
    # df_cat = pd.melt(df_cat).groupby(['variable', 'value']).size().to_frame(name='total')

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(data=df_cat, kind="count", x="variable", hue="value", col="cardio")

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df.loc[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) &
                     (df['height'] <= df['height'].quantile(0.975)) &
                     (df['weight'] >= df['weight'].quantile(0.025)) &
                     (df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, cbar_kws={"shrink": 0.5},
                center=0.0, annot=True, fmt=".1f", linewidths=.5, vmax=0.32, vmin=-0.16)

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
