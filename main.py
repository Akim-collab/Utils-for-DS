import matplotlib as mpl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
import pandas as pd

def get_df_info(df: pd.DataFrame, thr: float = 0.05) -> pd.DataFrame:
    """
    Возвращает информацию о каждом столбце DataFrame.

    Аргументы:
    df -- DataFrame, для которого нужно получить информацию.
    thr -- пороговое значение, по умолчанию 0.05.

    Возвращает:
    DataFrame с информацией о каждом столбце:
        type -- тип данных в столбце.
        unique -- количество уникальных значений.
        nan_share -- доля пропущенных значений.
        zero_share -- доля нулевых значений.
        empty_share -- доля пустых значений (для строковых столбцов).
        vc_max -- наиболее часто встречающееся значение и его доля.
        example -- два случайных примера из столбца.
        trash_score -- оценка "мусорности" столбца.

    Оценка "мусорности" рассчитывается по формуле:
        trash_score = max(nan_share + zero_share + empty_share, vc_max)
        при условии, что vc_max > thr, иначе trash_score = 0.

    Спецсимволом "-1" обозначается чистый 0, чтобы не путать с округлившимся 0.00001
    """
    res_df = pd.DataFrame(index=df.columns)
    for col in df.columns:
        dtype = df[col].dtype.name

        n_unique = len(df[col].unique())

        n_nan = df[col].isna().sum()
        nan_share = round(n_nan / len(df[col]), 3)

        n_zeros = (df[col] == 0).sum()
        zero_share = round(n_zeros / len(df[col]), 3)

        n_empty = (df[col].astype(str) == '').sum()
        empty_share = round(n_empty / len(df[col]), 3)

        vc_max = df[col].value_counts(normalize=True).head(1)
        vc_max_str = f'{vc_max.index[0]} ({round(vc_max.values[0], 3)})' if not vc_max.empty else ''

        examples = list(df[col].dropna().unique())[:2]
        examples_str = ', '.join([str(x) for x in examples]) if examples else ''

        trash_score = max(nan_share + zero_share + empty_share,
                          vc_max.values[0] if not vc_max.empty and vc_max.values[0] > thr else 0)
        trash_score_str = round(trash_score, 3) if trash_score != 0 else '-1'

        res_df.loc[col, 'type'] = dtype
        res_df.loc[col, 'unique'] = n_unique
        res_df.loc[col, 'nan_share'] = nan_share if n_nan / len(df[col]) != 0 else '-1'
        res_df.loc[col, 'zero_share'] = zero_share if n_zeros / len(df[col]) != 0 else '-1'
        res_df.loc[col, 'empty_share'] = empty_share if n_empty / len(df[col]) != 0.0 else '-1'
        res_df.loc[col, 'vc_max'] = vc_max_str
        res_df.loc[col, 'example'] = examples_str
        res_df.loc[col, 'trash_score'] = trash_score_str

    res_df.index.name = 'column'

    return res_df

# настройка размера шрифта легенды
mpl.rcParams['legend.title_fontsize'] = 20
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 20

def plot_density(df: pd.DataFrame, hue: str, cols: List[str] = None, drop_zero: bool = False, max_cat_thr: int = 20):
    """
    Функция для отрисовки распределения колонок датафрейма.

    Аргументы:
    df -- датафрейм, содержащий данные для отрисовки
    cols -- список колонок для отрисовки
    hue -- колонка для раскраски
    drop_zero -- флаг, указывающий, нужно ли удалять нулевые значения перед отрисовкой (по умолчанию False)
    max_cat_thr -- порог для определения категориальных колонок

    Возвращает:
    None

    """

    if cols is None:
        cols = df.columns

    for column in cols:

        if column == hue or column not in df.select_dtypes(include=[np.number, object]).columns:
            continue

        if df[column].dtype == np.number:

            print('################################################################################')

            if drop_zero:
                data = df[df[column] != 0][[column, hue]]
            else:
                data = df[[column, hue]]

            fig, ax = plt.subplot_mosaic(mosaic='''abc''')

            sns.histplot(data=data, x=column, hue=hue, multiple='stack', element='step', stat='count',
                         alpha=0.8, ax=ax['a'])
            if drop_zero:
                ax['a'].set_title(f'histplot of {column} without zeros', fontsize=15)
            else:
                ax['a'].set_title(f'histplot of {column} with zeros', fontsize=15)

            sns.boxenplot(data=data, y=column, x=hue, hue=hue, showfliers=False,
                          order=df[hue].unique(), ax=ax['b'])
            ax['a'].set(xlabel=None)

            if drop_zero:
                ax['b'].set_title(f'boxenplot of {column} without zeros', fontsize=15)
            else:
                ax['b'].set_title(f'boxenplot of {column} with zeros', fontsize=15)

            ax['b'].set(xlabel=None, ylabel=None)
            ax['b'].get_legend().remove()

            sns.stripplot(data=data.sample(n=200), y=column, x=hue, hue=hue,
                          order=df[hue].unique(), size=3, ax=ax['c'])

            if drop_zero:
                ax['c'].set_title(f'stripplot of {column} without zeros', fontsize=15)
            else:
                ax['c'].set_title(f'stripplot of {column} with zeros', fontsize=15)

            ax['c'].set(xlabel=None, ylabel=None)
            ax['c'].get_legend().remove()

            fig.set_size_inches(30, 7)
            fig.suptitle(f'{column} vs {hue}', fontsize=25)

            plt.show()

            if df[column].isin([0, np.nan]).any():
                fig, ax = plt.subplots(figsize=(15, 9))

                data = df.groupby(hue)[column].agg([lambda x: x.isna().sum(), lambda x: (x == 0).sum()])
                data = data.rename(columns={'<lambda_0>': 'NaN', '<lambda_1>': '0'})
                data = data.reset_index()

                tidy = data.melt(id_vars=hue).rename(columns=str.title)

                tidy.loc[tidy['Value'] == 0, 'Value'] -= 0.1 * np.max(tidy['Value'])

                sns.barplot(data=tidy, x='Variable', y='Value', hue=hue, ax=ax)

                ax.get_legend().remove()
                ax.set(xlabel=None)
                ax.axhline(0, color='black', ls='--')
                ax.grid(True, axis='y')

                plt.show()

        elif len(df[column].unique()) <= max_cat_thr:

            print('################################################################################')

            data = df.replace('', 'missing')
            data = data.fillna('missing')

            data = data[[column, hue]]

            fig, ax = plt.subplots(figsize=(15, 9))

            sns.countplot(data=data, x=column, hue=hue, edgecolor='black', order=data[column].value_counts().index,
                          hue_order=data[hue].unique(), ax=ax)
            ax.set(xlabel=None)
            ax.set_title(f'straight hue', fontsize=15)
            ax.tick_params('x', rotation=90)
            ax.grid(True, axis='y')
            fig.suptitle(f'{column} vs {hue}', fontsize=25)
            plt.show()