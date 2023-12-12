import matplotlib.pyplot as plt
import champion_dicts
import numpy as np

def plot_history(history):
    # accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'])
    plt.show()

def best_performance(history, print_results=True) -> tuple:
    """Returns the best train and val accuracy and loss from a history object
    Returns:
        tuple: (best_train_acc, best_val_acc, best_train_loss, best_val_loss)
        Each tuple is of the form (value, epoch)
    """
    best_train_acc = (-1, -1)
    best_val_acc = (-1, -1)

    best_train_loss = (1e10, -1)
    best_val_loss = (1e10, -1)

    for i in range(len(history.history['accuracy'])):
        if history.history['accuracy'][i] > best_train_acc[0]:
            best_train_acc = (history.history['accuracy'][i], i)
        if history.history['val_accuracy'][i] > best_val_acc[0]:
            best_val_acc = (history.history['val_accuracy'][i], i)
        if history.history['loss'][i] < best_train_loss[0]:
            best_train_loss = (history.history['loss'][i], i)
        if history.history['val_loss'][i] < best_val_loss[0]:
            best_val_loss = (history.history['val_loss'][i], i)

    if print_results:
        print("Best train accuracy: ", best_train_acc)
        print("Best val accuracy: ", best_val_acc)
        print("Best train loss: ", best_train_loss)
        print("Best val loss: ", best_val_loss)
    
    return best_train_acc, best_val_acc, best_train_loss, best_val_loss

def print_summary(hist, model_name=None):
    """Prints a summary of the model and the best train and val accuracy and loss
    Args:
        hist: history object from model.fit
        model_name: name of the model to print
    """
    if model_name is not None:
        print("Model: ", model_name)
    print("Model summary: ")
    
    train_acc, val_acc, train_loss, val_loss = best_performance(hist)
    print(f"Best train accuracy: {train_acc[0]:.4f} at epoch {train_acc[1]}")
    print(f"Best val accuracy: {val_acc[0]:.4f} at epoch {val_acc[1]}")
    print(f"Best train loss: {train_loss[0]:.4f} at epoch {train_loss[1]}")
    print(f"Best val loss: {val_loss[0]:.4f} at epoch {val_loss[1]}")

class ModelStats:
    def __init__(self, history, test_acc_loss = (0, 999), name="Unnamed model"):

        self.history = history
        self.name = name
        self.test_acc = test_acc_loss[1]
        self.test_loss = test_acc_loss[0]
        self.measurements = []

class ModelComparator:
    def __init__(self, test_data):
        self.models = []
        self.test_x = test_data[0]
        self.test_y = test_data[1]

        self.measurements = []

    def add_measurement(self, name, func):
        self.measurements.append((name, func))

    def add_model(self, model, history, name="Unnamed model"):
        """Adds a model to the comparator
        Args:
            model: the model to add
            history: the history object from model.fit
            test_x: the test data
            test_y: the test labels
            name: the name of the model
        """
        print("Evaluating ", name)
        test_acc_loss = model.evaluate(self.test_x, self.test_y)
        print("Test accuracy: ", test_acc_loss[1])
        print("Test loss: ", test_acc_loss[0])

        # remove model with the same name
        for i in range(len(self.models)):
            if self.models[i].name == name:
                self.models.pop(i)
                break

        m_stats = ModelStats(history, test_acc_loss, name)

        # calc other measurements
        for m_name, m_func in self.measurements:
            y = m_func(model)
            m_stats.measurements.append((m_name, y))

        self.models.append(m_stats)

    def sort_by_test_acc(self):
        """Sorts the models by test accuracy"""
        self.models.sort(key=lambda x: x.test_acc, reverse=True)

    def sort_by_test_loss(self):
        """Sorts the models by test loss"""
        self.models.sort(key=lambda x: x.test_loss)

    def print_table(self, print_results=True):
        """
        Prints table of model stats as markdown
        """

        self.sort_by_test_acc()

        txt = "| Model | Test acc | Test loss |"
        for m_name, _ in self.measurements:
            txt += f" {m_name} |"
        txt += "\n"
        txt += "| --- | ---: | ---: |"
        for _ in self.measurements:
            txt += " ---: |"
        txt += "\n"

        for model in self.models:
            txt += f"| {model.name} | {model.test_acc:.4f} | {model.test_loss:.4f} |"
            for _, m in model.measurements:
                txt += f" {m:.4f} |"
            txt += "\n"

        if print_results:
            print(txt)
        return txt


    def print_summary(self):
        """Prints a summary of the models"""
        # sort by test accuracy
        self.sort_by_test_acc()
        for model in self.models:
            s = f"{model.name}: Test acc: {model.test_acc:.4f}, Test loss: {model.test_loss:.4f}"
            print(s)
    
    def plot_bar(self):
        """Plots a bar chart of the test accuracy and loss of the models"""
        self.sort_by_test_acc()
        names = [model.name for model in self.models]
        test_accs = [model.test_acc for model in self.models]
        test_losses = [model.test_loss for model in self.models]
        x = np.arange(len(names))
        width = 0.35
        fig, ax = plt.subplots()
        # plot accuracy
        rects1 = ax.bar(x - width/2, test_accs, width, label='Accuracy')

        ax.set_ylabel('Accuracy')
        ax.set_title('Test accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()

        fig.tight_layout()

        plt.show()

        # plot loss
        fig, ax = plt.subplots()
        # plot loss
        rects1 = ax.bar(x - width/2, test_losses, width, label='Loss')

        ax.set_ylabel('Loss')
        ax.set_title('Test loss')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()

        fig.tight_layout()

        plt.show()


    def plot_histories(self, val_only=True):
        """Plots a graph of the train and val accuracy and loss of the models"""

        # one graph for all losses, one for all accuracies
        def plot_by_key(key):
            names = []
            for model_stats in self.models:
                hist = model_stats.history
                val = hist.history[key]
                plt.plot(val)
                names.append(model_stats.name)

            plt.title('Model '+key)
            plt.ylabel(key)
            plt.xlabel('Epoch')
            plt.legend(names, loc='upper left')
            plt.show()

        for key in ["accuracy", "val_accuracy", "loss", "val_loss"]:
            if val_only and "val" not in key:
                continue
            plot_by_key(key)

        return

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
def visualize_embeddings(embedding_layer, size, k_clusters=10):
    """Visualizes the embeddings from an embedding layer as a 2d scatter plot
    Args:
        embedding_layer: the embedding layer to visualize
        size: the size of the vocabulary
        k_clusters: amount of clusters for k means clustering. Clusters are displayed in the same color
    """
    conv = champion_dicts.ChampionConverter()
    # get embeddings
    embeddings = embedding_layer.get_weights()[0]

    # K means clustering
    k = k_clusters
    kmeans = KMeans(n_clusters=k, n_init="auto")
    clusters = kmeans.fit_predict(embeddings)
    colors = np.random.rand(k, 3)
    # embeddings.shape = (vocab_size, embedding_dim)
    # reduce dimensions to 2 for visualization
    pca = PCA(n_components=2)
    embeddings = pca.fit_transform(embeddings)
    # embeddings.shape = (vocab_size, 2)
    # plot
    plt.figure(figsize=(10,10))
    for i in range(size):
        color = colors[clusters[i]]
        plt.scatter(embeddings[i,0], embeddings[i,1], color=color)
        name = "Unknown"
        try:
            name = conv.get_champion_name_from_index(i)
        except:
            pass
        plt.annotate(name, (embeddings[i,0], embeddings[i,1]))
    plt.show()


def print_embedding_norms(embedding_layer, size, n_closest=10):
    conv = champion_dicts.ChampionConverter()
    # get embeddings
    embeddings = embedding_layer.get_weights()[0]

    smallest = []

    norm_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i==j:
                norm_matrix[i,j] = 999999
                continue

            if i>j:
                norm_matrix[i, j] = norm_matrix[j, i]
                continue

            em1 = embeddings[i]
            em2 = embeddings[j]

            norm = np.linalg.norm(em1-em2)
            norm_matrix[i, j] = norm

            if len(smallest)<n_closest:
                smallest.append((float(norm), i, j))
            elif smallest[n_closest-1][0]>float(norm):
                smallest.pop()
                smallest.append((float(norm), i, j))
            else:
                continue

            smallest.sort(key=lambda x:x[0])

    print("Closest champion embeddings:")
    for i in smallest:
        norm = round(i[0], 4)
        name1 = "Unknown"
        name2 = "Unknown"
        try:
            name1 = conv.get_champion_name_from_index(i[1])
        except:
            pass
        try:
            name2 = conv.get_champion_name_from_index(i[2])
        except:
            pass
        print(f"{name1} and {name2}: {norm}")

    




def export_embeddings(embedding_layer, size, path):
    """Exports the embeddings from an embedding layer to a file
    Args:
        embedding_layer: the embedding layer to export
        size: the size of the vocabulary
        path: the path to save the embeddings
    """
    conv = champion_dicts.ChampionConverter()
    # write vectors
    f = open(path+"_data.tsv", "w")
    for i in range(size):
        f.write("\t".join([str(x) for x in embedding_layer.get_weights()[0][i]]) + "\n")
    f.close()

    # write metadata
    f = open(path+"_metadata.tsv", "w")
    for i in range(size):
        name = "Unknown"
        try:
            name = conv.get_champion_name_from_index(i)
        except:
            pass
        f.write(name + "\n")
    f.close()

if __name__ == "__main__":
    # create fake data
    import tensorflow as tf

    embedding = tf.keras.layers.Embedding(10, 5)
    embedding.build((None,))
    labels = tf.constant([0,1,2,3,4,5,6,7,8,9])
    print_embedding_norms(embedding, 10)
    visualize_embeddings(embedding, 10)