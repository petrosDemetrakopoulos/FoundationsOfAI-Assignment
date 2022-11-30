import numpy as np
import matplotlib.pyplot as plt

categories = ['WINS', 'DRAWS','LOSES']

data = {"7": [2,0,8],
        "8.5": [4,0,6],
        "10": [6,2,2]}

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
        print(str(widths))
        text_color = 'darkgrey'
        if color == 'Red' or 'Yellow':
            text_color = 'black'
        else:
            text_color = 'white'

        #ax.bar_label(rects,fontsize=16, label_type='center', color=text_color)
    for i,c in enumerate(ax.containers):
        print(i)
        print(c)
        print(c.datavalues)
        labels = [v if v > 0 else "" for v in c.datavalues]
        text_color = 'darkgrey'
        if i == 0 or i == 2:
            text_color = 'white'
        else:
            text_color = 'black'
        ax.bar_label(c, labels=labels, fontsize=16, label_type='center', color=text_color)

    ax.legend(ncol=len(category_names), bbox_to_anchor=(1, 1),
              loc='lower right', fontsize='medium')
    ax.set_title('Agent vs Greedy Player on empty-4x4', fontsize=18)
    ax.set_ylabel('Time limit', fontsize=18)
    ax.set_xlabel('Frequency of outcome')
    ax.set_yticklabels([7,8.5,10], fontsize=18)
    return fig, ax

plot_chart(data, categories)

plt.show()
