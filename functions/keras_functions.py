"""
# This is a set of machine learning tools developed using Python for Keras
"""

import eli5
from eli5.sklearn import PermutationImportance
from sklearn.feature_selection import SelectFromModel

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras import optimizers
from keras.constraints import maxnorm
from keras.optimizers import Adam, Nadam, RMSprop, SGD, Adadelta, Adamax
from keras.losses import logcosh, binary_crossentropy
from keras.activations import relu, elu

from keras.wrappers.scikit_learn import KerasClassifier
from decorators import time_function 


# Set a wrapper function for neural networks
@time_function
def neural_network_function_wrapper(
    num_neurons_
    , num_hidden_layers_
    , input_dim_
    , kernel_initializer_
    , activation_
    , kernel_constraint_
    , dropout_rate_
    , output_kernel_initializer_
    , output_activation_
    , loss_
    , optimizer_
    , learning_rate_
    , momentum_
    , rho_
    , beta_1_
    , beta_2_
    , return_metrics_
):

    def neural_network_function(
        num_neurons=num_neurons_
        , num_hidden_layers=num_hidden_layers_
        , input_dim=input_dim_
        , kernel_initializer=kernel_initializer_
        , activation=activation_
        , kernel_constraint=kernel_constraint_
        , dropout_rate=dropout_rate_
        , output_kernel_initializer=output_kernel_initializer_
        , output_activation=output_activation_
        , loss=loss_
        , optimizer=optimizer_
        , learning_rate=learning_rate_
        , momentum=momentum_
        , rho=rho_
        , beta_1=beta_1_
        , beta_2=beta_2_
        , return_metrics=return_metrics_
    ):
        model = Sequential()

    # Specify the first layer
        model.add(Dense(num_neurons, input_dim=input_dim, kernel_initializer=kernel_initializer, activation=activation, kernel_constraint=maxnorm(kernel_constraint)))
        model.add(Dropout(dropout_rate))

    # Add more hidden layers
        for i in range(num_hidden_layers-1):       
            model.add(Dense(num_neurons, kernel_initializer=kernel_initializer, activation=activation, kernel_constraint=maxnorm(kernel_constraint)))
            model.add(Dropout(dropout_rate))

    # Specify the output layer
        model.add(Dense(1, kernel_initializer=output_kernel_initializer, activation=output_activation))

        if optimizer=='SGD':
            optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
        elif optimizer=='Adadelta':
            optimizer = Adadelta(learning_rate=learning_rate, rho=rho)
        elif optimizer=='Adam':
            optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        elif optimizer=='Nadam':
            optimizer = Nadam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        elif optimizer=='Adamax':
            optimizer = Adamax(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        else:
            print('ERROR: The optimizer has not been set up')

        if return_metrics=='True':
            model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
        else: 
            model.compile(loss=loss, optimizer=optimizer)
        return model
    
    return neural_network_function

# Select the top features from a neural network using a specific threshold
@time_function
def top_keras_feat(
    dev_df # Input data frame name
    , feats # List of features that will be used in the neural network
    , target # Target variable name
    , weight_variable # Weight variable name
    , threshold # Threshold above which the list of variables will be returned
    , keras_function # Keras function wrapper that will be used to create the neural network
    , epochs_ # Number of epochs for neural network
    , batch_size_ # Batch size for neural network
    , feat_importance_num_display # Number of feature importances to display the variable importance
    ):
   
    model = KerasClassifier(build_fn=keras_function, verbose=0)
    model.fit(dev_df[feats].values, dev_df[target], sample_weight=dev_df[weight_variable].values, use_multiprocessing=True, workers=8, verbose=0,
              epochs=epochs_
              , batch_size=batch_size_)

    perm = PermutationImportance(model, random_state=1).fit(dev_df[feats].values, dev_df[target], 
                                                            sample_weight=dev_df[weight_variable].values)
#    Display the feature importances
    display(eli5.show_weights(perm, feature_names = dev_df[feats].columns.tolist(), top=feat_importance_num_display))

    select_from_model = SelectFromModel(perm, threshold=threshold, prefit=True)
    select_index = list(select_from_model.get_support(indices=True))
    
    return list(dev_df[feats].columns[i] for i in select_index)




