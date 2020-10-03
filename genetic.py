# Genetic algorithm for MNIST database.

# Import modules
import numpy as np
from network import init, init2
import random
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting
from collections import Counter
from sklearn.metrics import confusion_matrix
import itertools
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def mutate(new_individual, layers, mutate_factor):

    for i in layers:
        for bias in range(len(new_individual.layers[i].get_weights()[1])):
            n = random.random()
            if(n < mutate_factor):
                new_individual.layers[i].get_weights()[1][bias] *= random.uniform(-0.5, 0.5)

    for i in layers:
        for weight in new_individual.layers[i].get_weights()[0]:
            n = random.random()
            if(n < mutate_factor):
                for j in range(len(weight)):
                    if(random.random() < mutate_factor):
                        new_individual.layers[i].get_weights()[0][j] *= random.uniform(-0.5, 0.5)


    return new_individual
    

def crossover(individuals, layers, mutate_factor):
    no_of_individuals = len(individuals)
    new_individuals = []

    new_individuals.append(individuals[0])
    new_individuals.append(individuals[1])

    for i in range(2, no_of_individuals):
        if(i < (no_of_individuals - 2)):
            if(i == 2):
                parentA = random.choice(individuals[:3])
                parentB = random.choice(individuals[:3])
            else:
                parentA = random.choice(individuals[:])
                parentB = random.choice(individuals[:])

            for i in layers:
                temp = parentA.layers[i].get_weights()[1]
                parentA.layers[i].get_weights()[1] = parentB.layers[i].get_weights()[1]
                parentB.layers[i].get_weights()[1] = temp

                new_individual = random.choice([parentA, parentB])
            
        else:
             new_individual = random.choice(individuals[:])

        new_individuals.append(mutate(new_individual, layers, mutate_factor))
        #new_individuals.append(new_individual)

    return new_individuals


def evolve(individuals, losses, layers, mutate_factor):
    sorted_y_idx_list = sorted(range(len(losses)),key=lambda x:losses[x])
    individuals = [individuals[i] for i in sorted_y_idx_list ]

    #winners = individuals[:6]

    new_individuals = crossover(individuals, layers, mutate_factor)

    return new_individuals


def train(models, epochs, X_train, X_val, Y_train, Y_val):
    losses = []
    histories = []
    for i in range(len(models)):
        history = models[i].fit(x=X_train,y=Y_train, epochs=epochs, validation_data=(X_val, Y_val))
        losses.append(round(history.history['loss'][-1], 4))
        histories.append(history)
    return models, losses, histories


def train_and_load(models, epochs, X_train, X_val, Y_train, Y_val):
    losses = []
    histories = []
    for i in range(len(models)):
        model_i = keras.models.load_model(models[i])
        history = model_i.fit(x=X_train,y=Y_train, epochs=epochs, validation_data=(X_val, Y_val))
        losses.append(round(history.history['loss'][-1], 4))
        histories.append(history)
        model_i.save(models[i])
    return models, losses, histories


def train2(models, epochs, X_train, X_val, Y_train, Y_val, datagen, batch_size, callbacks):
    losses = []
    histories = []
    for i in range(len(models)):
        history = models[i].fit(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                epochs=epochs, validation_data=(X_val, Y_val),
                                verbose=1, steps_per_epoch=X_train.shape[0] // batch_size,
                                callbacks=callbacks)
        losses.append(round(history.history['loss'][-1], 4))
        histories.append(history)
    return models, losses, histories


#Note, this code is taken straight from the SKLEARN website, an nice way of viewing confusion matrix.
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    plt.figure()
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1
    plt.show()

def main():
    # Parameters
    no_of_generations = 20
    no_of_individuals = 10
    mutate_factor = 0.05
    layers = [0, 3, 5]
    batch_size = 64
    num_classes = 10
    epochs = 1
    all_models_inmemory = True
    use_datagenerator = False


    # Load dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    # Data normalization
    X_train = X_train.astype('float32')/255.0
    X_test = X_test.astype('float32')/255.0
    y_train_df = pd.DataFrame(data=y_train, columns=["label"])
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    # See some statistics
    print(y_train_df.head())
    z_train = Counter(y_train_df['label'])
    print(z_train)
    sns.countplot(y_train_df['label'])
    plt.show()
    # Preview the images first
    plt.figure(figsize=(12, 10))
    x, y = 10, 4
    for i in range(40):
        plt.subplot(y, x, i+1)
        plt.imshow(X_train[i],interpolation='nearest')
    plt.show()
    # Printing shapes
    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_train.shape[0], 'test samples')
    # Database splitting
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


    # Callbacks
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.0001)
    callbacks = [learning_rate_reduction]


    # Datagenerator
    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    datagen.fit(X_train)


    # Training
    if use_datagenerator:
        individuals = []
        for i in range(no_of_individuals):
            if all_models_inmemory:
                individuals.append(init2(num_classes, batch_size, epochs))
            else:
                individual = init2(num_classes, batch_size, epochs)
                name_ind = 'indv{}.h5'.format(i)
                individual.save(name_ind)


        for generation in range(no_of_generations):
            if generation == no_of_generations-1:
                individuals, losses, histories = train2(individuals, epochs, X_train, X_val, Y_train, Y_val, datagen, batch_size, callbacks)
                print(losses)
            else:
                individuals, losses, histories = train2(individuals, epochs, X_train, X_val, Y_train, Y_val, datagen, batch_size, callbacks)
                print(losses)
                individuals = evolve(individuals, losses, layers, mutate_factor)
    else:
        if all_models_inmemory:
            individuals = []
            for i in range(no_of_individuals):
                individuals.append(init(num_classes, batch_size, epochs))

            for generation in range(no_of_generations):
                if generation == no_of_generations - 1:
                    individuals, losses, histories = train(individuals, epochs, X_train, X_val, Y_train, Y_val)
                    print(losses)
                else:
                    individuals, losses, histories = train(individuals, epochs, X_train, X_val, Y_train, Y_val)
                    print(losses)
                    individuals = evolve(individuals, losses, layers, mutate_factor)
        else:
            individuals = []
            for i in range(no_of_individuals):
                individual = init(num_classes, batch_size, epochs)
                name_ind = 'indv{}.h5'.format(i)
                individuals.append(name_ind)
                individual.save(name_ind)

            for generation in range(no_of_generations):
                if generation == no_of_generations - 1:
                    individuals, losses, histories = train_and_load(individuals, epochs, X_train, X_val, Y_train, Y_val)
                    print(losses)
                else:
                    individuals, losses, histories = train_and_load(individuals, epochs, X_train, X_val, Y_train, Y_val)
                    print(losses)
                    individuals = evolve_and_load(individuals, losses, layers, mutate_factor)



    # Evaluation
    # Best individual
    ib = np.argmax(losses)
    final_loss, final_acc = individuals[ib].evaluate(X_val, Y_val, verbose=0)
    print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))

    # Confusion matrix
    # Predict the values from the validation dataset
    Y_pred = individuals[ib].predict(X_val)
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis = 1)
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(Y_val, axis = 1)
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    # plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, classes = range(10))

    # Plot learning curves
    print(histories[ib].history.keys())
    accuracy = histories[ib].history['accuracy']
    val_accuracy = histories[ib].history['val_accuracy']
    loss = histories[ib].history['loss']
    val_loss = histories[ib].history['val_loss']
    epochs = range(len(accuracy))
    plt.figure()
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


    # Errors are difference between predicted labels and true labels
    errors = (Y_pred_classes - Y_true != 0)
    Y_pred_classes_errors = Y_pred_classes[errors]
    Y_pred_errors = Y_pred[errors]
    Y_true_errors = Y_true[errors]
    X_val_errors = X_val[errors]


    # Probabilities of the wrong predicted numbers
    Y_pred_errors_prob = np.max(Y_pred_errors, axis = 1)
    # Predicted probabilities of the true values in the error set
    true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
    # Difference between the probability of the predicted label and the true label
    delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
    # Sorted list of the delta prob errors
    sorted_dela_errors = np.argsort(delta_pred_true_errors)
    # Top 6 errors
    most_important_errors = sorted_dela_errors[-6:]
    # Show the top 6 errors
    display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)


    # Activations
    # It looks like diversity of the similar patterns present on multiple classes effect the performance of the classifier although CNN is a robust architechture.
    test_im = X_train[154]
    plt.imshow(test_im.reshape(28,28), cmap='viridis', interpolation='none')
    # Let's see the activation of the 2nd channel of the first layer:
    # Had taken help from the keras docs, this answer on StackOverFlow
    layer_outputs = [layer.output for layer in individuals[ib].layers[:8]]
    activation_model = tf.keras.models.Model(inputs=individuals[ib].inputs, outputs=layer_outputs)
    activations = activation_model.predict(test_im.reshape(1,28,28,1))
    first_layer_activation = activations[0]
    plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
    test_im = X_train[154]
    plt.imshow(test_im.reshape(28,28), cmap='viridis', interpolation='none')
    # Let's plot the activations of the other conv layers as well.
    individuals[ib].layers[:-1]# Droping The Last Dense Layer
    layer_names = []
    for layer in individuals[ib].layers[:-1]:
        layer_names.append(layer.name)
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        if layer_name.startswith('conv'):
            n_features = layer_activation.shape[-1]
            size = layer_activation.shape[1]
            n_cols = n_features // images_per_row
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_image = layer_activation[0,:, :, col * images_per_row + row]
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size : (col + 1) * size,
                                 row * size : (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()

    layer_names = []
    for layer in individuals[ib].layers[:-1]:
        layer_names.append(layer.name)
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        if layer_name.startswith('max'):
            n_features = layer_activation.shape[-1]
            size = layer_activation.shape[1]
            n_cols = n_features // images_per_row
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_image = layer_activation[0,:, :, col * images_per_row + row]
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size : (col + 1) * size,
                                 row * size : (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()

    '''
    layer_names = []
    for layer in individuals[ib].layers[:-1]:
        layer_names.append(layer.name)
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        if layer_name.startswith('drop'):
            n_features = layer_activation.shape[-1]
            size = layer_activation.shape[1]
            n_cols = n_features // images_per_row
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_image = layer_activation[0,:, :, col * images_per_row + row]
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size : (col + 1) * size,
                                 row * size : (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
    '''

    # Classification report
    # Predict the values from the validation dataset
    Y_pred = individuals[ib].predict(X_val)
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis = 1)
    Y_true_classes = np.argmax(Y_val, axis = 1)
    Y_pred_classes[:5], Y_true_classes[:5]
    target_names = ["Class {}".format(i) for i in range(num_classes)]
    print(classification_report(Y_true_classes, Y_pred_classes, target_names=target_names))

    # Predict the values from the test dataset
    Y_pred = individuals[ib].predict(X_test)
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis = 1)
    Y_true_classes = np.argmax(y_test, axis = 1)
    Y_pred_classes[:5], Y_true_classes[:5]
    target_names = ["Class {}".format(i) for i in range(num_classes)]
    print(classification_report(Y_true_classes, Y_pred_classes, target_names=target_names))


    # Save best model
    individuals[ib].save("cnn.h5")
    json_string = individuals[ib].to_json()


if __name__ == "__main__":
    main()
