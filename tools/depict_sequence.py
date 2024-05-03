import matplotlib.pyplot as plt

class DepictSequence:
    def __init__(self, data_list, titles, figure_directory=None):
        self.data_list = data_list
        self.titles = titles
        self.dir = figure_directory
        self.colors = ["red", "blue", "orange", "green", "purple", "brown", "pink", "gray", "olive", "cyan"]
        
    def display(self, title=None, show_last_seqeunce=-1):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12),
                                 dpi=80, facecolor="w", edgecolor="k", num=title)
        fig.suptitle(title, fontsize=16)
        self.drawAxis(axes[0, 0], self.data_list[0], 'train')
        self.drawAxis(axes[0, 1], self.data_list[1], 'val')
        
        if show_last_seqeunce > 0:
            self.drawAxis(axes[1, 0], self.data_list[0], 'train', start=-show_last_seqeunce)
            self.drawAxis(axes[1, 1], self.data_list[1], 'val', start=-show_last_seqeunce)
        else:
            fig.delaxes(axes[1, 0])
            fig.delaxes(axes[1, 1])
            
        plt.show()

    def drawAxis(self, axis, data, title, start=0, length=-1.0):
        for j in range(len(data)):
            # determine the range for drawing
            s = start
            e = len(data[j])
            if start < 0:
                s = e + start
            elif length > 0 and start + length < e:
                e = start + length
            x = range(s, e)        

            # if j==0:
            #     plt.plot(x, data[j][s:e], color=colors[j % (len(colors))], marker='*', label=titles[i][j])
            # else:
            axis.plot(x, data[j][s:e], color=self.colors[j % (len(self.colors))], label=self.titles[j])
        axis.legend(loc='lower right')
        axis.set_title(title)

    # def display(self, title=None, start=0, length=-1.0, select=[0,1,2]):
    #     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6),
    #                              dpi=80, facecolor="w", edgecolor="k", num=title)
    #     fig.suptitle(title, fontsize=16)
    #     axes[0].set_title('train')
    #     axes[1].set_title('val')
    #     for i in range(len(self.data_list)):
    #         c = self.colors[i % (len(self.colors))]
    #         data = self.data_list[i]

    #         # show the entire sequence by default
    #         s = start
    #         e = len(data[0]) - 1

    #         if start < 0:
    #             s = e + start
    #         elif length > 0 and start + length < e:
    #             e = start + length
    #         x = range(s, e)
    #         plt.subplot(1, 2, i+1)
    #         # plt.xticks(x)
    #         # for j in range(len(data)):
    #         for j in [k for k in select if k < len(data)]:
    #             # if j==0:
    #             #     plt.plot(x, data[j][s:e], color=colors[j % (len(colors))], marker='*', label=titles[i][j])
    #             # else:
    #             plt.plot(x, data[j][s:e], color=self.colors[j % (len(self.colors))], label=self.titles[j])
    #         plt.legend(loc='lower right')
    #     # plt.tight_layout()
    #     plt.show()
