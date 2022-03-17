import time
import numpy as np
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
import seaborn as sns
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import matplotlib.pyplot as plt


def plot_computation_graph(graph,
                             # figsize=(15, 15),
                             scaling_factor=1,
                             width_scaling_factor=1,
                             height_scaling_factor=1,
                             node_color_palette='pastel',
                             node_size=150,
                             edge_alpha=0.5,
                             edge_width=0.5,
                             font_size=8,
                             title=None,
                             show_fig=True,
                             dpi=600,
                             verbose=False):
    start_time = time.time_ns()

    # fig = plt.figure(figsize=figsize)
    aesthetics = PlotAesthetics()
    aesthetics.set_icml_paper_plot_aesthetics(dpi=dpi)
    fig = plt.figure(figsize=aesthetics.get_standard_fig_size(scaling_factor=scaling_factor, width_scaling_factor=width_scaling_factor, height_scaling_factor=height_scaling_factor))
    
    pos = graphviz_layout(graph, prog='dot')
    
    node_labels = {node: node for node in graph.nodes}
    nx.draw_networkx_nodes(graph,
                           pos,
                           node_size=node_size,
                           node_color=sns.color_palette(node_color_palette)[0],
                           label=node_labels)
    
    nx.draw_networkx_edges(graph,
                           pos,
                           alpha=edge_alpha,
                           width=edge_width)
    
    nx.draw_networkx_labels(graph, 
                            pos, 
                            labels=node_labels,
                            font_size=font_size)
    
    if title is not None:
        plt.title(title)
    
    if show_fig:
        plt.show()
    
    if verbose:
        print(f'Constructed figure in {(time.time_ns() - start_time) * 1e-9:.3f} s')

    return fig


class PlotAesthetics:
    def __init__(self):
        pass

    def set_icml_paper_plot_aesthetics(self,
                                      context='paper',
                                      style='ticks',
                                      linewidth=0.75,
                                      font_scale=1,
                                      palette='colorblind',
                                      desat=1,
                                      dpi=300):
        
        # record params
        self.context = context
        self.linewidth = linewidth
        self.font_scale = font_scale
        self.palette = palette
        self.desat = desat
        self.dpi = dpi

        # apply plot config
        sns.set(rc={'text.usetex': True,
                    'figure.dpi': dpi,
                    'savefig.dpi': dpi})
        sns.set(font="times")
        sns.set_theme(font_scale=font_scale,
                      context=context,
                      style=style,
                      palette=palette)

        
    def get_standard_fig_size(self,
                              col_width=3.25, 
                              col_spacing=0.25, 
                              n_cols=1,
                              scaling_factor=1,
                              width_scaling_factor=1,
                              height_scaling_factor=1):
        
        # save params
        self.col_width = col_width
        self.col_spacing = col_spacing
        self.n_cols = n_cols
        self.scaling_factor=scaling_factor
        self.width_scaling_factor = width_scaling_factor
        self.height_scaling_factor = height_scaling_factor
    
        # calc fig size
        self.fig_width = ((col_width * n_cols) + ((n_cols - 1) * col_spacing))
        golden_mean = (np.sqrt(5) - 1.0) / 2.0    # Aesthetic ratio
        self.fig_height = self.fig_width * golden_mean
        return (scaling_factor * width_scaling_factor * self.fig_width, scaling_factor * height_scaling_factor * self.fig_height)

    def get_winner_bar_fig_size(self,
                                col_width=3.25, 
                                col_spacing=0.25, 
                                n_cols=1):
        # save params
        self.col_width = col_width
        self.col_spacing = col_spacing
        self.n_cols = n_cols

        # calc fig size
        self.fig_width = ((col_width * n_cols) + ((n_cols - 1) * col_spacing))
        self.fig_height = self.fig_width * 1.25

        return (self.fig_width, self.fig_height)





def get_plot_params_config(font_size):
    params = {'legend.fontsize': font_size*0.75,
              'axes.labelsize': font_size,
              'axes.titlesize': font_size,
              'xtick.labelsize': font_size*0.75,
              'ytick.labelsize': font_size*0.75}
    return params
