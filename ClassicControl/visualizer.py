import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    data = clean('CartPoleResultsv1.csv')
    plot(data)
    plt.savefig('CartPoleResultsv1.png')


def clean(inputPath: str, episodes: int) -> pd.DataFrame:
    df = pd.read_csv(inputPath, index_col=0)
    names = []
    scores = []
    for column in df.columns:
        names += re.findall("([0-9].[0-9])", column) * episodes
        scores += list(df[column])
    new_df = pd.DataFrame()
    new_df['Version'] = names
    new_df['Scores'] = scores
    return new_df


def plot(data: pd.DataFrame):
    sns.catplot(x='Scores', y='Version', data=data, palette=sns.color_palette('flare'), aspect=2.5)


if __name__ == '__main__':
    main()