import matplotlib.pyplot as plt

class DepictHistory:
    def __init__(self, model_history, figure_directory=None):
        self.history = model_history.history
        self.epochs = model_history.epoch
        self.dir = figure_directory

    def display(self, title=None):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6),
                                 dpi=80, facecolor="w", edgecolor="k", num=title)
        fig.suptitle(title, fontsize=16)

        epochs = self.epochs
        keys = self.history.keys()
        if 'loss' in keys:
            loss = self.history['loss']
            val_loss = self.history['val_loss']
            axis = axes[0]
            # plt.subplot(1, 2, 1)
            # plt.xticks(epochs)
            axis.plot(epochs, loss, 'r', label='Training Loss')
            axis.plot(epochs, val_loss, 'bo', label='Validation Loss')
            # plt.ylim([0, 1])
            axis.hlines(min(val_loss), 0, len(epochs) - 1, colors=['b'], ls="--", lw=1.0)
            axis.legend(loc='upper right')
            # plt.title('(%i, %i) Training and Validation Loss' % (num_images, BATCH_SIZE))
        else:
            fig.delaxes(axes[0])

        if 'accuracy' in keys:
            acc = self.history['accuracy']
            val_acc = self.history['val_accuracy']
            axis = axes[1]
            # plt.subplot(1, 2, 2)
            # plt.xticks(epochs)
            axis.plot(epochs, acc, 'r', label='Training Accuracy')
            axis.plot(epochs, val_acc, 'bo', label='Validation Accuracy')
            # plt.ylim([0.7, 1.0])
            axis.hlines(max(val_acc), 0, len(epochs)-1, colors=['b'], ls="--", lw=1.0)
            axis.legend(loc='lower right')
            # plt.title('(%i, %i) Training and Validation Accuracy' % (num_images, BATCH_SIZE))
        else:
            fig.delaxes(axes[1])

        if self.dir == None:
            plt.show()
        else:
            fig.savefig(self.dir+"Results.png", bbox_inches='tight')
            plt.close()

    # def display(self, title=None):
    #     epochs = self.epochs
    #     keys = self.history.keys()
    #     if ('loss' in keys) & ('accuracy' in keys):
    #         fig = plt.figure(figsize=(12, 6), num=title)
    #         ax1 = fig.add_subplot(121)
    #         ax2 = fig.add_subplot(122)

    #         loss = self.history['loss']
    #         val_loss = self.history['val_loss']
    #         # plt.subplot(1, 2, 1)
    #         # plt.xticks(epochs)
    #         ax1.plot(epochs, loss, 'r', label='Training Loss')
    #         ax1.plot(epochs, val_loss, 'bo', label='Validation Loss')
    #         # plt.ylim([0, 1])
    #         ax1.hlines(min(val_loss), 0, len(epochs) - 1, colors=['b'], ls="--", lw=1.0)
    #         ax1.legend(loc='upper right')
    #         # plt.title('(%i, %i) Training and Validation Loss' % (num_images, BATCH_SIZE))

    #         acc = self.history['accuracy']
    #         val_acc = self.history['val_accuracy']
    #         # plt.subplot(1, 2, 2)
    #         # plt.xticks(epochs)
    #         ax2.plot(epochs, acc, 'r', label='Training Accuracy')
    #         ax2.plot(epochs, val_acc, 'bo', label='Validation Accuracy')
    #         # plt.ylim([0.7, 1.0])
    #         ax2.hlines(max(val_acc), 0, len(epochs)-1, colors=['b'], ls="--", lw=1.0)
    #         ax2.legend(loc='lower right')
    #         # plt.title('(%i, %i) Training and Validation Accuracy' % (num_images, BATCH_SIZE))
    #     elif 'loss' in keys:
    #         fig = plt.figure(figsize=(6, 6), num=title)
    #         loss = self.history['loss']
    #         val_loss = self.history['val_loss']
    #         # plt.subplot(1, 1, 1)
    #         # plt.xticks(epochs)
    #         plt.plot(epochs, loss, 'r', label='Training Loss')
    #         plt.plot(epochs, val_loss, 'bo', label='Validation Loss')
    #         # plt.ylim([0, 1])
    #         plt.hlines(min(val_loss), 0, len(epochs) - 1, colors=['b'], ls="--", lw=1.0)
    #         plt.legend(loc='upper right')
    #         # plt.title('(%i, %i) Training and Validation Loss' % (num_images, BATCH_SIZE))
    #     elif 'accuracy' in keys:
    #         fig = plt.figure(figsize=(6, 6), num=title)
    #         acc = self.history['accuracy']
    #         val_acc = self.history['val_accuracy']
    #         # plt.subplot(1, 2, 2)
    #         # plt.xticks(epochs)
    #         plt.plot(epochs, acc, 'r', label='Training Accuracy')
    #         plt.plot(epochs, val_acc, 'bo', label='Validation Accuracy')
    #         # plt.ylim([0.7, 1.0])
    #         plt.hlines(max(val_acc), 0, len(epochs)-1, colors=['b'], ls="--", lw=1.0)
    #         plt.legend(loc='lower right')
    #         # plt.title('(%i, %i) Training and Validation Accuracy' % (num_images, BATCH_SIZE))

    #     if self.dir == None:
    #         plt.show()
    #     else:
    #         fig.savefig(self.dir+"Results.png", bbox_inches='tight')
    #         plt.close()

