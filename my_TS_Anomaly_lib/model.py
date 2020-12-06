from tensorflow.keras.layers import Input, LSTM, Dropout \
                                    , RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam


#/////////////////////////////////////////////////////////////////////////////////////


# The autoencoder network model constructor
def autoencoder_model(
    batch_size = None
    , time_steps = None
    , features_count = None
):
    """
    Define a 'build_model' function that can be
    passed to a Keras-Tuner tuner
    (i.e. a function that takes an HParams parameter).
    """
    def build_model(hp):

        # encoder head
        inputs = Input(batch_size = None, shape=(time_steps, features_count)
                       , name='encoder_head')

        # parameterized number of layers :
        num_layers = hp.Int('num_layers', 0, 3)
        # lstm layers parameters :
        lstm_units = hp.Int('lstm_units', min_value = 4, max_value = 290, sampling="log")

        # encoder body
        previous_layer = inputs
        for i in range(num_layers) :
            lstm_layer = LSTM(2*(num_layers-i)*lstm_units, activation='relu'
                              , name='encoder_lstm_'+str(i)
                              , return_sequences=True
                             )(previous_layer)
            previous_layer = lstm_layer

        # encoder tail
        L2 = LSTM(lstm_units, activation='relu'
                  , name='encoder_lstm_tail'
                  , return_sequences=False
                 )(previous_layer)
        dropout_rate = hp.Float('dropout_rate', min_value = 0.01, max_value = 0.2, sampling="linear")
        L3 = Dropout(rate=dropout_rate
                     , name='encoder_dropout'
                    )(L2)
        L4 = RepeatVector(time_steps
                          , name='encoder_tail'
                         )(L3)

        # decoder head
        L5 = LSTM(lstm_units, activation='relu'
                  , name='decoder_lstm_head'
                  , return_sequences=True
                 )(L4)

        # decoder body
        previous_layer = L5
        for i in range(num_layers) :
            lstm_layer = LSTM(2*(i+1)*lstm_units, activation='relu'
                              , name='decoder_lstm_'+str(i)
                              , return_sequences=True
                             )(previous_layer)
            previous_layer = lstm_layer

        # decoder tail
        L7 = Dropout(rate=dropout_rate
                     , name='decoder_dropout'
                    )(previous_layer)
        output = TimeDistributed(Dense(features_count
                                       , name='dense'
                                      )
                                 , name='decoder_tail'
                                )(L7)

        model = Model(inputs=inputs, outputs=output)

        # DO NOT allow too large values or risk encountering gradient exploiding :
        learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5])
        model.compile(
            optimizer = Adam(learning_rate)
            , loss = 'mean_squared_error' # 'mae' # 'mean_squared_logarithmic_error' # 
        )

        return model

    return build_model


#/////////////////////////////////////////////////////////////////////////////////////


import numpy as np
import os, cv2

from tensorflow.keras.utils import model_to_dot


def plot_model(
    keras_model
    , bgcolor, forecolor
    , framecolor, watermark_text
    , fillcolors, fontcolors
    , show_layer_names = False
) -> np.ndarray :
    """
    Parameters :
        - keras_model (keras.engine.training.Model) :
            the model the architecture of which is to be plotted.
        - bgcolor (str) :
            the hexadecimal expression of the background color.
        - forecolor (str) :
            the hexadecimal expression of the foreground color.
        - framecolor (str) :
            the hexadecimal expression of the color
            of the image border.
        - watermark_text (str) :
            the text to be watermarked at the bottom-right corner
            of the image.
        - fillcolors (dic) :
            a dictionnary with entries ('node name': 'fill hex color')
            for nodes to be filled with a color other than transparent.
        - fontcolors (dic) :
            a dictionnary with entries ('node name': 'font hex color')
            for nodes to be labell in a color other than 'forecolor'.

    Resuts :
        - an 'np.ndarray' of 3 colors channels
          and dimension (height x width)
    """

    graph = model_to_dot(keras_model, show_layer_names=show_layer_names)
    graph.set_bgcolor(bgcolor)

    nodes = graph.get_node_list() ; edges = graph.get_edge_list()

    for node in nodes:
        if node.get_name() == 'node' :
            node.obj_dict['attributes'] = \
                {'shape': 'record', 'style': "filled, rounded"
                 , 'fillcolor': bgcolor, 'color': forecolor
                 , 'fontcolor': forecolor, 'fontname': 'helvetica'}
        node_label = node.get_label()
        #print(str(node_label).partition(": ")[0] + " - " + str((node is not None) & (node_label in fillcolors)))
        if node_label is not None :
            layer = keras_model.get_layer(str(node_label).partition(": ")[0].partition("(")[0])
            # HTML-style graphviz label (between "<>" signs)
            node.set_label(
                "<" +
                node_label +
                (
                    "<BR />&nbsp;" +
                     "<FONT POINT-SIZE=\"12\">return_sequences = " + str((layer.return_sequences)) + "</FONT>"
                     if isinstance(layer, LSTM) else ""
                ) +
                (
                    "<BR />&nbsp;" +
                     "<FONT POINT-SIZE=\"10\">" + str(layer.output_shape) + "</FONT>"
                     if not isinstance(layer, Dropout) else ""
                ) +
                ">"
            )
            if isinstance(layer, Dropout) :
                node.set_height(.0) # remove extra spacing / margin
                node.set_style("dashed, rounded")
                node.set_fontsize(11)

            #print(layer.name)
            if layer.name in fillcolors :
                node.set_fillcolor(fillcolors[layer.name])
            if layer.name in fontcolors :
                node.set_fontcolor(fontcolors[layer.name])

    # 'graph.set_edge_defaults' will fail (defaults being appended last) =>
    for edge in edges: # apply successively to each edge
        edge.set_color(forecolor) ; edge.set_arrowhead("vee")

    graph.set_ranksep(.3) # <- spacing between nodes of different ranks
    graph.set_nodesep(.3) # <- spacing between nodes of different ranks
    #print(graph.to_string())
    os.makedirs(os.path.join('.', 'tmp'), exist_ok=True)
    tmp_filename = os.path.join('tmp', 'colored_tree.png')
    graph.write_png(tmp_filename)
    #from IPython.core.display import display, Image ; display(Image(graph.create_png()))

    image = cv2.cvtColor(cv2.imread(tmp_filename), cv2.COLOR_BGR2RGB)
    os.remove(tmp_filename)

    def hex_to_rgb(hex_color) : return tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    #watermark
    bottom_pad = 20
    texted_image = cv2.putText(
        img =
            cv2.copyMakeBorder(image, 0, 20, 0, 0, borderType=cv2.BORDER_CONSTANT, value=hex_to_rgb(bgcolor))
        , text=watermark_text
        , org=(
            image.shape[1]-(10+int(7.33*len(watermark_text)))
            , image.shape[0]+bottom_pad-10
        )
        , fontFace=cv2.FONT_HERSHEY_COMPLEX
        , fontScale=.4, color=hex_to_rgb(framecolor), lineType=cv2.LINE_AA, thickness=1
    )
    # padding
    outer_pad_top, outer_pad_bot, outer_pad_left, outer_pad_right = 4, 4, 4, 4 ; outer_pad_color = hex_to_rgb(bgcolor)
    inner_pad_top, inner_pad_bot, inner_pad_left, inner_pad_right = 2, 2, 2, 2 ; inner_pad_color = hex_to_rgb(framecolor)
    padded_image = cv2.copyMakeBorder(
        cv2.copyMakeBorder(texted_image
                           , inner_pad_top, inner_pad_bot, inner_pad_left, inner_pad_right
                           , borderType=cv2.BORDER_CONSTANT, value=inner_pad_color)
        , outer_pad_top, outer_pad_bot, outer_pad_left, outer_pad_right
        , borderType=cv2.BORDER_CONSTANT, value=outer_pad_color)

    return cv2.cvtColor(padded_image, cv2.COLOR_RGB2BGR)


def plot_autoencoder(best_model) :
    """
    Formatted model plot compatible with a
    model built via the 'autoencoder_model.build_model' method.

    Resuts :
        - an 'np.ndarray' of 3 colors channels
    """

    num_layers = int(len(best_model.layers) - 7 / 2)

    architecture_plot = plot_model(
        best_model
        , bgcolor = '#123456', forecolor = '#a18a12'
        , framecolor = '#e6c619', watermark_text = "Time Series / Anomaly Detection"
        , fillcolors = {
            'encoder_head': '#2b5acf'
            , **dict([('encoder_lstm_'+str(i), '#1f74c2') for i in range(num_layers)])
            , 'encoder_lstm_tail': '#f58940'
            , 'encoder_tail': '#f58940', 'decoder_lstm_head': '#bf9000'
            , **dict([('decoder_lstm_'+str(i), '#ffdb97') for i in range(num_layers)])
            , 'decoder_tail': '#ffd580'}
        , fontcolors = {
            'encoder_head': '#01206e'
            , **dict([('encoder_lstm_'+str(i), '#012580') for i in range(num_layers)])
            , 'encoder_lstm_tail': '#753001', 'encoder_tail': '#753001'
            , 'decoder_lstm_head': '#362301'}
        , show_layer_names = True
    )

    return architecture_plot


#/////////////////////////////////////////////////////////////////////////////////////


from kerastuner import HyperParameters

def rebuild_model(hyperparameters_dict) :
    """
    Taking advantage of the fact that
    we can pass a hyperparameters argument 'hp'
    to the tuner constructor.

    Parameters :
        - hyperparameters_dict (dict) :
            dictionnary of hyperparameters values

    Results :
        - (tensorflow.python.keras.engine.functional.Functional)
            a compiled but untrained autoencoder model.
    """
    hp = HyperParameters()
    for hparam in hyperparameters_dict :
        hp.Fixed(hparam, value=hyperparameters_dict[hparam])

    #print(hp._space)

    return autoencoder_model(time_steps = X_train.shape[1]
                             , features_count = X_train.shape[2]
                            )(hp)


#/////////////////////////////////////////////////////////////////////////////////////


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_predictions(
    test_data
    , scored_test
) -> None :
    """
    Plots two timelines :
        - the reconstructed signal loss (with detected anomaly 'y/n' highlight)
        - the raw signal (with detected anomaly 'y/n' overlay).
    """

    assert test_data.columns[0] == 'timestamp' and len(test_data.columns) == 2 \
           , '"test_data" input dataframe structure exception'
    assert {'Loss_mae', 'Threshold', 'Anomaly'}.issubset(scored_test.columns) \
           , '"scored_test" input dataframe structure exception'

    sensor_name = test_data.columns[1]

    fig, ax = plt.subplots(nrows=2, sharex = 'col', figsize=(15, 9))

    ax[0].plot(test_data['timestamp']
               , scored_test[['Loss_mae']]
               , color = 'purple'
              )
    ax[0].plot(test_data['timestamp']
               , scored_test[['Threshold']]
               , label = 'Anomaly Threshod'
               , color = 'orange', alpha=.66
              )
    ax[0].legend(['Loss_mae', 'Threshold'], loc='upper left')
    ax_ = ax[0].twinx()
    ax_.plot(test_data['timestamp']
             , scored_test['Anomaly'], label = 'anomaly detected'
             , color = 'g', linewidth=.4
            )
    ax_.set_ylabel('Signal is abnormal y/n', rotation = -90, labelpad=20)
    ax_.set_yticks([0, 1]) ; ax_.set_yticklabels(['No', 'Yes'])
    ax_.legend(loc='upper right')
    ax[0].xaxis.grid(True, linestyle='--', dashes=(5, 10), which='minor', alpha=.4)
    ax[0].xaxis.grid(True, linestyle='--', dashes=(5, 5), which='major', alpha=.5)
    ax[0].axhline(y=0, color='k', zorder=0, linewidth=.4)
    ax[0].set_yscale('log') ; ax[0].set_ylabel('Signal Reconstruction Error (log scale)')

    ax[1].plot(test_data['timestamp']
               , test_data[sensor_name]
           )
    ax[1].scatter(test_data['timestamp']
                  , test_data[sensor_name]
                   , c=['r' if bool_ else 'g' for bool_ in scored_test['Anomaly']]
                   , s=[20 if bool_ else 0 for bool_ in scored_test['Anomaly']]
                   , linewidth=0
                  )
    ax[1].set_ylabel('Vibration Amplitude\n(one-second average absolute)')
    ax[1].minorticks_on() ; ax[1].tick_params(which='minor', length=0)
    ax[1].tick_params(axis='x', which='major', length=10, color='b', width=.5)
    ax[1].xaxis.grid(True, linestyle='--', dashes=(5, 10), which='minor', alpha=.4)
    ax[1].xaxis.grid(True, linestyle='--', dashes=(5, 5), which='major', alpha=.5)
    ax[1].set_xlabel('timestamp')
    ax[1].spines['bottom'].set_position('zero')
    
    myFmt = mdates.DateFormatter('%Y-%m-%d %Hh') ; ax[1].xaxis.set_major_formatter(myFmt)
    for tick in ax[1].get_xticklabels() : tick.set_rotation(45) ; tick.set_ha('right')

    plt.show()


#/////////////////////////////////////////////////////////////////////////////////////
































































