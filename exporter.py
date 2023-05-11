import os
import pandas as pd

class Exporter:
    def __init__(self, image_fn = str, plot_data = list, legend_data = list):
        self.image_fn = image_fn                
        self.plot_data = plot_data                  # data extracted using PlotOCR class
        self.legend_data = legend_data              # data extracted using PlotOCR class

    # renormalizes x-axis of data extracted by plot
    def renormalize_x_group(self, value, norm_min, norm_max):
        x_values = [lb.center[0] for lb in self.plot_data]
        
        min_x = min(x_values)
        max_x = max(x_values)
        
        norm_value = (norm_max - norm_min) * (value - min_x)/(max_x - min_x)
        
        return norm_value

    # renormalizes x-axis of data extracted by plot
    def renormalize_y_stake(self, value, norm_min, norm_max):
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
    def construct_dataset(self):
        datas = []

        for lb in self.plot_data:
            kind = None
            for legb in self.legend_data:
                if legb.color == lb.color_rgb:
                    kind = legb.label
            group_value = self.renormalize_x_group(lb.center[0], 0, 300)
            stake_value = self.renormalize_y_stake(lb.center[1], 0, 300)
            
            datas.append((lb.label, group_value, stake_value, kind))

        df = pd.DataFrame(datas, columns=('Label', 'GroupRel', 'StakeRel', 'Kind'))

        print('Showing the first rows of the dataset:')
        print(df.head())

        img_extension = self.image_fn[-3:len(self.image_fn)]

        out_dir = os.path.join('out')

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        csv_fn = self.image_fn.replace(img_extension, 'csv').replace('src/', '')
        out_path = os.path.join(out_dir, csv_fn)

        df.to_csv(out_path)
        print('\nExtracted data was exported to {fn}'.format(fn = out_path))


