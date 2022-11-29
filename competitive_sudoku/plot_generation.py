import numpy as np
import matplotlib.pyplot as plt

categories = ['WINS', 'DRAWS','LOSES']

data = {"0.1": [0,0,10],
        "0.5": [0,0,10],
        "1": [0,0,10],
        "5": [2,1,7]}

def plot_chart(results, category_names):
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = ['Green','Yellow','Red']

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        text_color = 'darkgrey'
        if color == 'Red':
            text_color = 'black'
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(1, 1),
              loc='lower right', fontsize='small')
    ax.set_title('Agent vs Greedy Player on random-4x4')
    ax.set_ylabel('Time limit')
    ax.set_xlabel('Frequency of outcome')
    return fig, ax

plot_chart(data, categories)
plt.show()
