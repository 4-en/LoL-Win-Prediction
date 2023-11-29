import matplotlib.pyplot as plt

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