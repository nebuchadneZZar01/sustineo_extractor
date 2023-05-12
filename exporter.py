import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

class Exporter:
    def __init__(self, image_fn = str, plot_data = list, legend_data = list):
        self.__image_fn = image_fn                
        self.__plot_data = plot_data                  # data extracted using PlotOCR class
        self.__legend_data = legend_data              # data extracted using PlotOCR class

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
    def compose_dataset(self):
        datas = []

        for lb in self.plot_data:
            kind = None
            for legb in self.legend_data:
                if legb.color == lb.color_rgb:
                    kind = legb.label
            group_value = self.__renormalize_x_group(lb.center[0], 0, 300)
            stake_value = self.__renormalize_y_stake(lb.center[1], 0, 300)
            
            datas.append((lb.label, group_value, stake_value, kind))

        self.__dataframe = pd.DataFrame(datas, columns=('Label', 'GroupRel', 'StakeRel', 'Kind'))
        
        # removing last blank char
        self.__dataframe['Label'] = self.__dataframe.apply(lambda x: x['Label'][:-1], axis = 1)
        
        print('Showing the first rows of the dataset:')
        print(self.__dataframe.head())

    def export_dataset(self):
        img_extension = self.__image_fn[-3:len(self.__image_fn)]

        out_dir = os.path.join('out', 'csv')

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        csv_fn = self.__image_fn.replace(img_extension, 'csv').replace('src/', '')
        out_path = os.path.join(out_dir, csv_fn)

        self.__dataframe.to_csv(out_path)
        print('\nExtracted data was exported to {fn}'.format(fn = out_path))

    def compose_export_dataset(self):
        self.compose_dataset()
        self.export_dataset()

    def compose_export_plot(self):
        n_points = len(self.dataframe)

        fig_width = max(7, n_points / 2)
        fig_height = max(8, n_points / 4)

        label_colors = {label: color for label, color in zip(self.dataframe['Label'].unique(), cm.tab20.colors)}
        kind_colors = {kind: color for kind, color in zip(self.dataframe['Kind'].unique(), cm.tab20.colors)}

        label_mapping = {label: i for i, label in enumerate(self.dataframe['Label'].unique())}
        kind_mapping = {kind: i for i, kind in enumerate(self.dataframe['Kind'].unique())}

        fig, ax = plt.subplots(figsize=(fig_width,fig_height))

        scatter = ax.scatter(self.dataframe['GroupRel'], self.dataframe['StakeRel'], c=[label_mapping[label] for label in self.dataframe['Label']], edgecolors=[kind_colors[kind] for kind in self.dataframe['Kind']], cmap='tab20')

        label_cmap = cm.get_cmap('tab20', len(self.dataframe['Label'].unique()))
        label_legend_elements = [Patch(facecolor=label_cmap(i), label=label) for i, label in enumerate(self.dataframe['Label'].unique())]

        kind_cmap = cm.get_cmap('tab20', len(self.dataframe['Kind'].unique()))
        kind_legend_elements = [Patch(facecolor=kind_cmap(i), edgecolor=kind_cmap(i), label=kind) for i, kind in enumerate(self.dataframe['Kind'].unique())]

        ax.legend(handles=label_legend_elements, title='Label', loc = 'lower left', bbox_to_anchor=(0, 1))

        ax2 = ax.twinx()
        ax2.legend(handles=kind_legend_elements, title='Kind', loc = 'lower right', bbox_to_anchor=(1, 1))
        ax2.axis('off')

        ax.set_xlabel('Group Relevance')
        ax.set_ylabel('Stakeholders Relevance')
        fig.tight_layout()

        # EXPORTATION SECTION
        img_extension = self.__image_fn[-3:len(self.__image_fn)]

        out_dir = os.path.join('out', 'img')

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        img_fn = self.__image_fn.replace(img_extension, 'png').replace('src/', '')
        out_path = os.path.join(out_dir, img_fn)

        plt.savefig(out_path)
        plt.show()

        print('\nConverted image was exported to {fn}'.format(fn = out_path))