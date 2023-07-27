import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

class Exporter:
    def __init__(self, image_fn = str, plot_data = list, legend_data = list):
        self.__image_path = image_fn
        self.__image_basename = os.path.basename(image_fn)
        self.__plot_data = plot_data                            # data extracted using PlotOCR class
        self.__legend_data = legend_data                        # data extracted using PlotOCR class

        self.__dataframe = None

    @property
    def plot_data(self):
        return self.__plot_data

    @property
    def legend_data(self):
        return self.__legend_data

    @property
    def dataframe(self):
        return self.__dataframe

    # renormalizes x-axis of data extracted by plot
    def __renormalize_x_group(self, value, norm_min, norm_max):
        x_values = [lb.center[0] for lb in self.plot_data]
        
        min_x = min(x_values)
        max_x = max(x_values)
        
        norm_value = (norm_max - norm_min) * (value - min_x)/(max_x - min_x)
        
        return norm_value

    # renormalizes x-axis of data extracted by plot
    def __renormalize_y_stake(self, value, norm_min, norm_max):
        y_values = [lb.center[1] for lb in self.plot_data]

        min_y = min(y_values)
        max_y = max(y_values)

        norm_value = (norm_max - norm_min) * (value - min_y)/(max_y - min_y) 

        norm_value -= norm_max
        if norm_value < 0:
            norm_value *= -1

        return norm_value

    # function that makes a pandas dataframe
    # containing the extracted data
    def __compose_dataset(self):
        datas = []

        for lb in self.plot_data:
            macrotheme = 'Unknown'
            if self.legend_data is not None:
                for legb in self.legend_data:
                    if legb.color == lb.color_rgb:
                        macrotheme = legb.label
            group_value = self.__renormalize_x_group(lb.center[0], 0, 300)
            stake_value = self.__renormalize_y_stake(lb.center[1], 0, 300)
            
            datas.append((lb.label, macrotheme, group_value, stake_value))

        self.__dataframe = pd.DataFrame(datas, columns=('Label', 'Macrotheme', 'GroupRel', 'StakeRel'))
        
        # removing last blank char
        self.__dataframe['Label'] = self.__dataframe.apply(lambda x: x['Label'][:-1] if x['Label'][-1] == ' ' else x['Label'], axis = 1)
        
        print('Showing the first rows of the dataset:')
        print(self.__dataframe.head())

    # ranking by x, group relevance
    def __rank_by_x(self):
        df = self.dataframe
        df['RankGroup'] = df['GroupRel'].rank(method='min', ascending=True).astype(int)

        tmp = df['RankGroup'].values
        tmp_norm = np.interp(tmp, (tmp.min(), tmp.max()), (0,10))

        df['RankGroup'] = tmp_norm
    
    # ranking by y, stakeholders relevance
    def __rank_by_y(self):
        df = self.dataframe
        df['RankStake'] = df['StakeRel'].rank(method='min', ascending=True).astype(int)

        tmp = df['RankStake'].values
        tmp_norm = np.interp(tmp, (tmp.min(), tmp.max()), (0,10))

        df['RankStake'] = tmp_norm

    # ranking by (x,y), both group and stakeholders
    def __rank_absolute(self):
        df = self.dataframe
        df['RankAbsolute'] = df[['GroupRel', 'StakeRel']].apply(tuple, axis=1).rank(method='min', ascending=True).astype(int)

        tmp = df['RankAbsolute'].values
        tmp_norm = np.interp(tmp, (tmp.min(), tmp.max()), (0,10))

        df['RankAbsolute'] = tmp_norm

    def __add_ranking(self):
        self.__rank_by_x()
        self.__rank_by_y()
        self.__rank_absolute()

        print('\nAdded rankings by group, stakeholders and absolute relevances:')
        print(self.dataframe.head())

    # calculates alignment measure
    # using perpendicular distance
    # from the line passing through origin
    def __alignment(self, point):
        # y = ax + by + c
        # being the diagonal line we have the following values
        a = 1
        b = -1
        c = 0

        dist = abs((a*point[0]) + (b*point[1]) + c)/np.sqrt(a**2 + b**2)
        
        return dist

    def __add_alignment(self):
        df = self.dataframe
        df['Alignment'] = df[['GroupRel', 'StakeRel']].apply(tuple, axis=1).apply(self.__alignment)

        # normalization of the distance in [0, 100]
        # tipically these values are no higher than 300
        tmp = df['Alignment'].values
        tmp_norm = np.interp(tmp, (tmp.min(), tmp.max()), (0, 100))

        df['Alignment'] = tmp_norm

        print('\nAdded alignment measures:')
        print(self.dataframe.head())

    # exports csv format dataset
    def __export_dataset(self):
        img_extension = self.__image_basename[-3:len(self.__image_basename)]

        out_dir = os.path.join('out', 'csv')
        print(out_dir)

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        csv_fn = self.__image_basename.replace(img_extension, 'csv').replace('src/', '')
        out_path = os.path.join(out_dir, csv_fn)

        self.__dataframe.to_csv(out_path)
        print('\nExtracted data was exported to {fn}'.format(fn = out_path))

    def compose_export_dataset(self):
        self.__compose_dataset()
        self.__add_ranking()
        self.__add_alignment()
        self.__export_dataset()

    # draws distance between singular point and diagonal/alignment line
    def __draw_perpendicular_distance(self, axis, point_x, point_y):
        m_r = 1
        q_r = 0

        m_s = -1/m_r
        q_s = point_y - m_s * point_x

        x_perpendicular = (q_s - q_r) / (m_r - m_s)
        y_perpendicular = m_r * x_perpendicular

        axis.plot([point_x, x_perpendicular], [point_y, y_perpendicular], linestyle='--', color='orange', label='Alignment distance')

    # draws and exports new 
    # open-format png plot
    def compose_export_plot(self):
        n_points = len(self.dataframe)

        # size of the plot
        fig_width = max(8, n_points / 2)
        fig_height = max(8, n_points / 4)

        # taking colors for datasets
        macrotheme_colors = {macrotheme: color for macrotheme, color in zip(self.dataframe['Macrotheme'].unique(), cm.tab20.colors)}

        # mapping strings into numerical values
        label_mapping = {label: i for i, label in enumerate(self.dataframe['Label'].unique())}

        fig, ax = plt.subplots(figsize=(fig_width,fig_height))
        ax.grid(linestyle = '--')                               
        ax.axline((0,0), slope=1, linestyle='-', color='red', label='Perfect alignment')               # alignment line

        # generating the scatterplot
        scatter = ax.scatter(self.dataframe['GroupRel'], self.dataframe['StakeRel'], c=[label_mapping[label] for label in self.dataframe['Label']], edgecolors=[macrotheme_colors[macrotheme] for macrotheme in self.dataframe['Macrotheme']], cmap='tab20', marker='o', linewidth=2, s=[100 for el in range(len(label_mapping))])
        self.__draw_perpendicular_distance(ax, self.dataframe['GroupRel'], self.dataframe['StakeRel'])
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        # generating colormaps
        label_cmap = cm.get_cmap('tab20', len(self.dataframe['Label'].unique()))
        label_legend_elements = [Patch(facecolor=label_cmap(i), label=label) for i, label in enumerate(self.dataframe['Label'].unique())]

        macrotheme_cmap = cm.get_cmap('tab20', len(self.dataframe['Macrotheme'].unique()))
        macrotheme_legend_elements = [Patch(facecolor=macrotheme_cmap(i), edgecolor=macrotheme_cmap(i), label=macrotheme) for i, macrotheme in enumerate(self.dataframe['Macrotheme'].unique())]

        # # generating legends
        # line legend
        ax.legend(by_label.values(), by_label.keys())

        # label legend
        ax2 = ax.twinx()
        ax2.legend(handles=label_legend_elements, title='Label', loc = 'lower left', bbox_to_anchor=(0, 1))
        ax2.axis('off')

        # macrotheme legend
        ax3 = ax2.twinx()
        ax3.legend(handles=macrotheme_legend_elements, title='Macrotheme', loc = 'lower right', bbox_to_anchor=(1, 1))
        ax3.axis('off')

        ax.set_xlabel('Group Relevance')
        ax.set_ylabel('Stakeholders Relevance')
        fig.tight_layout()

        # EXPORTATION SECTION
        img_extension = self.__image_basename[-3:len(self.__image_basename)]

        out_dir = os.path.join('out', 'plot')

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        img_fn = self.__image_basename.replace(img_extension, 'png').replace('src/', '')
        out_path = os.path.join(out_dir, img_fn)

        plt.savefig(out_path)
        plt.show()

        print('Converted image was exported to {fn}\n'.format(fn = out_path))