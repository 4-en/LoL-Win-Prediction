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
        self.test_acc = test_acc_loss[0]
        self.test_loss = test_acc_loss[1]

class ModelComparator:
    def __init__(self):
        self.models = []

    def add_model(self, model, history, test_x, test_y, name="Unnamed model"):
        """Adds a model to the comparator
        Args:
            model: the model to add
            history: the history object from model.fit
            test_x: the test data
            test_y: the test labels
            name: the name of the model
        """
        test_acc_loss = model.evaluate(test_x, test_y)
        print("Test accuracy: ", test_acc_loss[0])
        print("Test loss: ", test_acc_loss[1])
        self.models.append(ModelStats(history, test_acc_loss, name))

    def sort_by_test_acc(self):
        """Sorts the models by test accuracy"""
        self.models.sort(key=lambda x: x.test_acc, reverse=True)

    def sort_by_test_loss(self):
        """Sorts the models by test loss"""
        self.models.sort(key=lambda x: x.test_loss)

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
        rects1 = ax.bar(x - width/2, test_accs, width, label='Test acc')
        rects2 = ax.bar(x + width/2, test_losses, width, label='Test loss')
        ax.set_ylabel('Accuracy/Loss')
        ax.set_title('Test accuracy and loss')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()

        fig.tight_layout()

        plt.show()

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
    kmeans = KMeans(n_clusters=k)
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
        plt.scatter(embeddings[i,0], embeddings[i,1], c=color)
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