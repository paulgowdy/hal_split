import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add, Conv2D, Dense, Flatten, Input,Lambda, Subtract, Concatenate, DepthwiseConv2D, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
import config as cfg

# ACTOR
# Takes [board, scalars, ship_select] as input
# Has seperate action heads for each possible ship of MAX_NB_SHIPS
# These get masked by the ship select layer
#tf.config.experimental_run_functions_eagerly(True)



def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        #print(y_true.shape)
        #print(y_pred.shape)

        #prob = K.sum(y_true * y_pred, axis=-1) #/ K.sum(y_true, axis=-1)
        #old_prob = K.sum(y_true * old_prediction, axis=-1) #/ K.sum(y_true, axis=-1)

        prob = K.sum(y_true * y_pred, axis=-1)
        old_prob = K.sum(y_true * old_prediction, axis=-1)

        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - cfg.LOSS_CLIPPING, max_value=1 + cfg.LOSS_CLIPPING) * advantage) + cfg.ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss


def create_actor_model():

    board_input = Input(shape=(cfg.BOARD_SIZE, cfg.BOARD_SIZE, 3))
    scalar_input = Input(shape=(2,))
    ship_select_input = Input(shape=(cfg.BOARD_SIZE, cfg.BOARD_SIZE, cfg.MAX_NB_SHIPS))

    ship_present = tf.keras.backend.sum(ship_select_input, 1)
    ship_present = tf.keras.backend.sum(ship_present, 1)
    # Should be just a list of 1s and 0s if there's a ship at that select layer or not...
    ship_present_extended = tf.keras.backend.repeat_elements(ship_present, cfg.NB_SHIP_ACTIONS, 0)

    advantages = Input(shape=(1,))
    old_predictions = Input(shape=(cfg.MAX_NB_SHIPS * cfg.NB_SHIP_ACTIONS,))
    #rewards = Input(shape=(1, 1,))
    #values = Input(shape=(1, 1,))
    # When I create branching output heads, I also need to create an "old_policy_probs" for each one
    # This will be needed to calculate the loss

    board_and_select = Concatenate(axis=-1)([board_input, ship_select_input])

    # Featurize Board and Scalars
    scalar_dense1 = Dense(4)(scalar_input)
    scalar_dense2 = Dense(4)(scalar_dense1)

    #conv1 = Conv2D(128, (3, 3), strides=1,  activation='relu', use_bias=True, padding="valid")(board_input)
    #conv2 = Conv2D(64, (3, 3), strides=1,  activation='relu', use_bias=True, padding="valid")(conv1)
    board_select_conv1 = DepthwiseConv2D(3, depth_multiplier=3, strides=1,  activation='relu', use_bias=True, padding="same")(board_and_select)
    board_select_conv2 = DepthwiseConv2D(3, depth_multiplier=3, strides=1,  activation='relu', use_bias=True, padding="same")(board_select_conv1)

    ship_output_heads = []

    for i in range(cfg.MAX_NB_SHIPS):
        #specific_ship_input = tf.gather_nd(ship_select_input, [[i]])
        ship_specific_combined = Concatenate(axis=-1)([board_select_conv2, tf.keras.backend.expand_dims(ship_select_input[:,:,:,i])])
        #ship_specific_combined = Concatenate(axis=-1, name='ConcatH{0}'.format(i))([board_select_conv2, specific_ship_input])

        ship_specific_conv1 = Conv2D(32, 3, strides=1, activation='relu', use_bias=True, padding="valid")(ship_specific_combined)
        ship_specific_conv2 = Conv2D(16, 3, strides=1, activation='relu', use_bias=True, padding="valid")(ship_specific_conv1)
        ship_specific_flat = Flatten()(ship_specific_conv2)

        ship_specific_with_scalar = Concatenate(axis=1)([ship_specific_flat, scalar_dense2])

        ship_specific_dense1 = Dense(16)(ship_specific_with_scalar)
        ship_specific_dense2 = Dense(16)(ship_specific_dense1)

        # Do the ship masking here
        #ship_specific_dense2_masked = Multiply()([ship_specific_orders, ship_present_extended[:,i]])

        # Also need to create "old_policy_probs" for each one for LOSS calculation

        ship_specific_orders = Dense(cfg.NB_SHIP_ACTIONS, activation='softmax')(ship_specific_dense2)
        ship_specific_orders_masked = Multiply()([ship_specific_orders[0,:], ship_present_extended[:,i]])

        #ship_output_heads.append(tf.keras.backend.expand_dims(ship_specific_orders_masked, axis=0))
        ship_output_heads.append(ship_specific_orders_masked)
        #ship_output_heads.append(ship_specific_combined)

    #ship_output_concat = Concatenate(axis=0)(ship_output_heads)
    ship_output_concat = tf.keras.backend.expand_dims(Concatenate(axis=-1)(ship_output_heads), axis=0)
    #ship_output_concat = tf.keras.backend.expand_dims(ship_output_concat, axis=0)

    model = Model(inputs=[board_input, scalar_input, ship_select_input, advantages, old_predictions], # Need to include the loss components here...
                  outputs=[ship_output_concat])

    #model.compile(optimizer=Adam(lr=1e-4), loss=tf.keras.losses.MeanSquaredError())
    model.compile(optimizer=Adam(lr=1e-4), loss=[proximal_policy_optimization_loss(
                                                advantage=advantages,
                                                old_prediction=old_predictions)])
    '''
    ppo_loss(
    oldpolicy_probs=oldpolicy_probs,
    advantages=advantages,
    rewards=rewards,
    values=values)
    '''

    model.summary()

    return model
