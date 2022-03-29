import time
import numpy as np
from sigfig import sigfig
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
import seaborn as sns
import pandas as pd
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import matplotlib.pyplot as plt

from typing import Any, Union


def get_plot_params_config(font_size):
    params = {'legend.fontsize': font_size*0.75,
              'axes.labelsize': font_size,
              'axes.titlesize': font_size,
              'xtick.labelsize': font_size*0.75,
              'ytick.labelsize': font_size*0.75}
    return params

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





def plot_hist(df: pd.DataFrame,
              x: str,
              hue: str,
              xlabel: str = None,
              ylabel: str = None,
              element: Union['bars', 'step', 'poly'] = 'bars',
              fill: bool = True,
              cumulative: bool = False,
              stat: Union['count', 'frequency', 'probability', 'percent', 'density'] = 'count',
              common_norm: bool = True,
              multiple: Union['layer', 'dodge', 'stack', 'fill'] = 'layer',
              xlog: bool = False,
              xaxis_label_style: str = 'plain', # 'plain' 'sci'
              scaling_factor: int = 1,
              width_scaling_factor: int = 1,
              height_scaling_factor: int = 1,
              title: str = None,
              show_fig: bool = True,
              palette: str = 'colorblind',
              dpi: int = 300):
    '''
    To plot a CDF, set:
        element = 'step'
        fill = False
        cumulative = True
        stat = 'density'
        common_norm = False
    '''
    aesthetics = PlotAesthetics()
    aesthetics.set_icml_paper_plot_aesthetics(palette=palette, dpi=dpi)

    f, ax = plt.subplots(figsize=aesthetics.get_standard_fig_size(scaling_factor=scaling_factor, width_scaling_factor=width_scaling_factor, height_scaling_factor=height_scaling_factor))
    g = sns.histplot(data=df,
                     x=x,
                     hue=hue,
                     element=element,
                     fill=fill,
                     cumulative=cumulative,
                     stat=stat,
                     common_norm=common_norm,
                     multiple=multiple,
                     log_scale=xlog)

    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(x)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.ticklabel_format(style=xaxis_label_style, axis='x', scilimits=(0,0))
    ax.tick_params(axis='both', which='major', pad=2)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    sns.despine(ax=ax) # remove top and right spines
    plt.gcf().patch.set_alpha(0.0)
    if show_fig:
        plt.show()
    return g






def plot_bar(df: pd.DataFrame,
             x: str,
             y: str,
             xlabel: str = None,
             ylabel: str = None,
             estimator: Any = np.mean,
             display_values: bool = True,
             display_values_y_offset: float = 0.15,
             ci: Union[float, 'sd', None] = None,
             errcolor: str = 'gray',
             capsize: float = 0.05,
             yaxis_label_style: str = 'plain', # 'plain' 'sci'
             scaling_factor: int = 1,
             width_scaling_factor: int = 1,
             height_scaling_factor: int = 1,
             title: str = None,
             show_fig: bool = True,
             palette: str = 'colorblind',
             dpi: int = 300):
    aesthetics = PlotAesthetics()
    aesthetics.set_icml_paper_plot_aesthetics(palette=palette, dpi=dpi)

    f, ax = plt.subplots(figsize=aesthetics.get_standard_fig_size(scaling_factor=scaling_factor, width_scaling_factor=width_scaling_factor, height_scaling_factor=height_scaling_factor))
    g = sns.barplot(data=df,
                    x=x,
                    y=y,
                    estimator=estimator,
                    ci=ci,
                    errcolor=errcolor,
                    capsize=capsize)

    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(x)
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(y)
    if title is not None:
        plt.title(title)
    plt.ticklabel_format(style=yaxis_label_style, axis='y', scilimits=(0,0))
    ax.tick_params(axis='both', which='major', pad=2)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    sns.despine(ax=ax) # remove top and right spines
    if display_values:
        show_values_on_bars(ax, sigfigs=3, y_offset=display_values_y_offset)
    plt.gcf().patch.set_alpha(0.0)
    if show_fig:
        plt.show()
    return g


def show_values_on_bars(axs, sigfigs=2, y_offset=0):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
#             _y = p.get_y() + p.get_height()
            _y = p.get_y() + y_offset
            value = sigfig.round(p.get_height(), sigfigs=min(sigfigs, len(str(int(p.get_height())))))
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)



def plot_line(df: pd.DataFrame,
              x: str,
              y: str,
              hue: str,
              xlabel: str = None,
              ylabel: str = None,
              xlim: list = None,
              ylim: list = None,
              xaxis_label_style: str = 'plain', # 'plain' 'sci'
              yaxis_label_style: str = 'plain', # 'plain' 'sci'
              xlog: bool = False,
              ylog: bool = False,
              ci: Union[int, 'sd', None] = 95,
              err_style: Union['band', 'bars'] = 'band',
              scaling_factor: int = 1,
              width_scaling_factor: int = 1,
              height_scaling_factor: int = 1,
              legend: bool = True,
              title: str = None,
              show_fig: bool = True,
              palette: str = 'colorblind',
              dpi: int = 300):
    aesthetics = PlotAesthetics()
    aesthetics.set_icml_paper_plot_aesthetics(palette=palette, dpi=dpi)

    f, ax = plt.subplots(figsize=aesthetics.get_standard_fig_size(scaling_factor=scaling_factor, width_scaling_factor=width_scaling_factor, height_scaling_factor=height_scaling_factor))
    g = sns.lineplot(data=df, 
                     x=x, 
                     y=y, 
                     hue=hue, 
                     ci=ci,
                     err_style=err_style,
                     linewidth=aesthetics.linewidth, 
                     legend=legend)

    if xlim is not None:
        plt.xlim(left=xlim[0], right=xlim[1])
    if ylim is not None:
        plt.ylim(bottom=ylim[0], top=ylim[1])
    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(x)
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(y)
    if title is not None:
        plt.title(title)
    plt.ticklabel_format(style=xaxis_label_style, axis='x', scilimits=(0,0))
    plt.ticklabel_format(style=yaxis_label_style, axis='y', scilimits=(0,0))
    ax.tick_params(axis='both', which='major', pad=2)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    sns.despine(ax=ax) # remove top and right spines
    if xlog:
        g.set(xscale='log')
    if ylog:
        g.set(yscale='log')
    plt.gcf().patch.set_alpha(0.0)
    if show_fig:
        plt.show()
    return g
















