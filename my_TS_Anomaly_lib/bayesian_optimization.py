import os, re, json
import pandas as pd, numpy as np
import matplotlib.pyplot as plt


#/////////////////////////////////////////////////////////////////////////////////////


import tensorflow
# ANIHILATE WARNING : Layer lstm will not use cuDNN kernel
# since it doesn't meet the cuDNN kernel criteria... whatever
tensorflow.get_logger().setLevel('ERROR')
tensorflow.autograph.set_verbosity(3)

from kerastuner.tuners import BayesianOptimization
from kerastuner.engine import tuner_utils
from tensorboard.plugins.hparams import api as hparams_api


class KerasTunerBayesianOptimization(BayesianOptimization):
    """
    Fix a bug in kerastuner that makes it impossible to work with TensorBoard subclassing
    (as TensorBoard in the original version is checked BY CLASS NAME !).
    [kerastuner.__version__ == 1.0.2]
    """
    def _configure_tensorboard_dir(self, callbacks, trial, execution=0) :
        """
        Overwritting the "_configure_tensorboard_dir" method of the
        "kerastuner.engine.multi_execution_tuner.MultiExecutionTuner" class
        from which "BayesianOptimization" inherits.
        """
        for callback in callbacks:
            #if callback.__class__.__name__ == 'TensorBoard': ### <<<=== THE ORGINAL FAULTY LINE <<<=== ###
            if isinstance(callback, TensorBoard):
                # Patch TensorBoard log_dir and add HParams KerasCallback
                logdir = self._get_tensorboard_dir(
                    callback.log_dir, trial.trial_id, execution)
                callback.log_dir = logdir
                hparams = tuner_utils.convert_hyperparams_to_hparams(
                    trial.hyperparameters)
                callbacks.append(
                    hparams_api.KerasCallback(
                        writer=logdir,
                        hparams=hparams,
                        trial_id=trial.trial_id))


#/////////////////////////////////////////////////////////////////////////////////////


from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K
import time


class TailoredTensorBoard(TensorBoard) :
    """
    Does additional 2 things compared to a standard TensorBoard Keras callback :
        - Add model.optimizer.lr value to the TensorBoard logged scalars
          (which is not done by default when a keras.callback altering it
           is used in conjonction with kerastuner).
        - Add epoch duration info to the TensorBoard logged scalars
          (in order to be able to see which model takes how long to train)
    """
    def __init__(self, log_dir, **kwargs) :
        super().__init__(log_dir=log_dir, **kwargs)

    def on_train_begin(self, logs=None) :
        self.startime = time.clock()

    #def on_test_begin(self, logs=None) :
    #    self.startime = time.clock()

    def on_epoch_end(self, epoch, logs=None) :
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        logs.update({'seconds': time.clock() - self.startime})
        super().on_epoch_end(epoch, logs)


#/////////////////////////////////////////////////////////////////////////////////////


from tensorboard.backend.event_processing import event_accumulator


def accumulator_to_pandas_df(
    events_filepath
    , scalar_names
) :
    """
    Retrieve the " measures from a Tensorboard event file.
    """
    ea = event_accumulator.EventAccumulator(
        events_filepath
        , size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 500
            , event_accumulator.IMAGES: 4
            , event_accumulator.AUDIO: 4
            , event_accumulator.SCALARS: 0
            , event_accumulator.HISTOGRAMS: 1
        })
    ea.Reload() # loads events from file

    scalar_names = [scalar_name for scalar_name in scalar_names
                    if scalar_name in ea.Tags()['scalars']]
    #print(scalar_names)
    if len(scalar_names) > 0 :
        result = None
        for scalar_name in scalar_names :
            if result is None :
                result = pd.DataFrame(ea.Scalars(scalar_name))
            else :
                result[scalar_name] = pd.DataFrame(ea.Scalars(scalar_name))['value']
        del ea
        return result
    else :
        del ea
        return None


#/////////////////////////////////////////////////////////////////////////////////////


def tuner_tensorboard_to_history_df(
    tensorboard_dir
    , verbose = 0
) :
    """
    Retrieve data generated by a callback of class 'TailoredTensorBoard'
    associated to a Keras Tuner of class ''.
    """

    scalar_names = ['epoch_loss', 'epoch_lr', 'epoch_seconds']
    history_df = pd.DataFrame(columns=['wall_time', 'trial_id', 'execution', 'step'
                                       , 'epoch_lr', 'epoch_seconds', 'train', 'valid'])
    for trial_id in os.listdir(tensorboard_dir) :
        # REMARK : As per the "os.listdir" documentation,
        # the list of events files is in arbitrary order !
        # (thus we retrieve the 'wall_time' TensorBoard Event attribute
        #  for chronological sorting purpose)

        if verbose : print(trial_id)
        folder_path = os.path.join(tensorboard_dir, trial_id)
        i = 0
        for execution in os.listdir(folder_path) :
            execution_history_df = pd.DataFrame(columns=['wall_time', 'trial_id', 'execution', 'step'
                                                         , 'epoch_lr', 'epoch_seconds', 'train'])
            if verbose : print(execution)
            ######################
            #  training history  #
            ######################
            for events_file in os.listdir(os.path.join(folder_path, execution, 'train')) :
                if re.match(r'events\.out\.tfevents\..*\.v2', events_file) :
                    if verbose : print('train: ' + events_file)
                    execution_train_history_df = \
                        accumulator_to_pandas_df(
                            os.path.join(folder_path, execution, 'train', events_file)
                            , scalar_names
                        ).rename(columns={'value':'train'})
                    #display(execution_train_history_df)
                    execution_history_df = pd.concat([
                        execution_history_df
                        , pd.concat([
                            execution_train_history_df
                            , pd.DataFrame(
                                {'trial_id': [trial_id]*execution_train_history_df.shape[0]
                                 , 'execution': [i]*execution_train_history_df.shape[0]}
                            )], axis = 1)
                        ], axis = 0).astype({"execution": int})
                    #display(execution_history_df)
            ######################
            # validation history #
            ######################
            for events_file in os.listdir(os.path.join(folder_path, execution, 'validation')) :
                if re.match(r'events\.out\.tfevents\..*\.v2', events_file) :
                    if verbose : print('valid: ' + events_file)
                    execution_history_df = pd.concat([
                        execution_history_df
                        , accumulator_to_pandas_df(
                            os.path.join(folder_path, execution, 'validation', events_file)
                            , [scalar_name for scalar_name in scalar_names
                               if scalar_name not in ['epoch_seconds', 'epoch_lr']]
                        ).rename(columns={'value':'valid'})['valid']
                    ], axis = 1)
                    #display(execution_history_df)

            history_df = pd.concat([
                history_df
                , execution_history_df
                ], axis = 0).astype({"execution": int})
            i += 1
    del execution_train_history_df, execution_history_df
    history_df = history_df.sort_values(by='wall_time').drop(['wall_time'], axis = 1).reset_index(drop=True)

    return history_df


#/////////////////////////////////////////////////////////////////////////////////////


def plot_tuner_models_histories(history_df) -> None :
    """
    Parameters :
        - history_df (pandas.DataFrame) :
            dataset as returned by
            the 'tuner_tensorboard_to_history_df' method

    Results :
        -N.A.
    """

    assert {'trial_id', 'execution', 'step', 'train', 'valid'}.issubset(history_df.columns) \
           , 'input dataframe structure exception'

    fig, ax = plt.subplots(ncols=2, sharey = 'row', figsize=(13, 4))

    # all trials (each trail averaged over its respective 'executions') :
    plot_data = \
        history_df.groupby(['trial_id', 'step'])[['train', 'valid']].mean().reset_index(
            ).pivot(index='step', columns=['trial_id'], values=['train', 'valid'])
    plot_data['train'].plot(ax=ax[0], color='lightgrey')
    plot_data['valid'].plot(ax=ax[1], color='lightgrey', linestyle="dotted")

    # tuner best_trial (also averaged over its 'executions') :
    best_trial_id = \
        history_df.groupby(['trial_id', 'step'])[['valid']].mean().idxmin()[0][0]
    plot_data = \
        plot_data.loc(axis=1)[pd.IndexSlice[:, best_trial_id]]
    plot_data['train'].plot(ax=ax[0], color='orange')
    plot_data['valid'].plot(ax=ax[1], color='orange', linestyle="dotted")

    ax[0].set_title("Training", y=0.9) ; ax[1].set_title("Validation", y=0.9)
    ax[0].set_xlabel('epochs') ; ax[1].set_xlabel('epochs')
    ax[0].set_yscale('log') ; ax[0].set_ylabel('validation loss (log scale)')
    ax[0].set_ylim([ax[0].get_ylim()[0], min(ax[0].get_ylim()[1], .5)])

    #Custom legend
    bestTrial = plt.Line2D((0,1),(0,0), color='orange', linestyle='dotted')
    otherTrials = plt.Line2D((0,1),(0,0), color='lightgrey', linestyle='dotted')
    ax[1].legend([bestTrial, otherTrials], ['best trial', 'others'])
    ax[0].get_legend().remove()

    fig.suptitle("Tuner {} trials, each averaged over their respective {} executions".format(
            len(history_df['trial_id'].unique()), history_df['execution'].max()+1
        ), fontsize=14)
    plt.tight_layout() ; fig.subplots_adjust(top=.92) ; plt.show() ; del plot_data


#/////////////////////////////////////////////////////////////////////////////////////


from datetime import timedelta

def plot_tuner_models_performances(history_df) -> None :
    """
    Plotted values, for each respective tuner trial,
    represent averages over all executions.

    BEWARE, for each tuner trial, we plot "average over last 10 epochs",
    which can result in one model being better ranked than
    the Tuner-identified "best model", since the Tuner ranks models
    based on respective (potentially even early) local
    "best (LOWEST EVER) val_loss".
    We highlight the "best" following Keras-Tuner definition
    (thus, may show as "not best" per the "last 10 epochs" rule)
    """

    best_trial_id = \
        history_df.groupby(['trial_id', 'step'])[['valid']].mean().idxmin()[0][0]
    ordered_trial_ids = history_df['trial_id'].unique()
    best_model_index = np.where(ordered_trial_ids == best_trial_id)[0][0]
    nb_epochs = history_df['step'].max() + 1

    # plotting average 'val_loss' over last 10 epochs =>
    fig, ax = plt.subplots(nrows = 2, sharex = 'col', figsize=(13, 5))
    plot_data = \
        history_df[history_df['step']>=nb_epochs-10].groupby(['trial_id', 'step'])[['train', 'valid']].mean() \
        .groupby(['trial_id']).min()['valid']
    plot_data = plot_data.loc[ordered_trial_ids] # <= chronologically re_ordered
    path_collection = ax[0].scatter(plot_data.index, plot_data.values, s=8**2, color='lightgrey')
    xy = np.delete(path_collection.get_offsets(), best_model_index, axis=0)
    path_collection.set_offsets(xy)
    ax[0].scatter(plot_data.index[best_model_index], plot_data.values[best_model_index], s=8**2, color='orange', alpha = .6)
    #ax.set_ylim([0,min(ax.get_ylim()[1], .5)])
    ax[0].set_yscale('log') ; ax[0].set_ylabel('validation loss* (log scale)')
    ax[0].text(0.01, .9, '*average over last 10 epochs'
               , transform=ax[0].transAxes, size=10, color='dimgrey')
    for label, x, y in zip(plot_data.values, plot_data.index, plot_data.values):
        ax[0].annotate(
            '{0:.3g}'.format(label),
            xy=(x, y), xytext=(-10, 10),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.05),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0', alpha=0.1)
            , color='dimgrey'
        )

    # plotting training duration =>
    duration_data = \
        history_df[history_df['step'] == nb_epochs-1].groupby(['trial_id'])['epoch_seconds'].mean()
    duration_data = duration_data.loc[ordered_trial_ids] # <= chronologically re_ordered
    bar = ax[1].bar(duration_data.index, duration_data.values, width = .4, color='lightgrey')
    bar.get_children()[best_model_index].remove()
    ax[1].bar(duration_data.index[best_model_index], duration_data.values[best_model_index]
              , width = .4, color='orange', alpha = .4)
    ax[1].set_xticks(ax[1].get_xticks())
    ax[1].set_xticklabels(range(len(ax[1].get_xticks())))
    ax[1].set_xlabel('tuner trial number')
    ax[1].set_ylabel('training duration (seconds)')

    for label, x, y in zip(duration_data.values, duration_data.index, duration_data.values):
        ax[1].annotate(
            str(timedelta(seconds=round(label))),
            xy=(x, y), xytext=(50, 10),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.05),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0', alpha=0.1)
            , color='dimgrey'
        )

    fig.suptitle('Bayesian Optimization - Keras Tuner Seach')
    plt.tight_layout() ; fig.subplots_adjust(top=.92) ; plt.show() ; del plot_data, duration_data


#/////////////////////////////////////////////////////////////////////////////////////


def tuner_trials_hparams_to_df(tuner_project_dir) :
    """
    When a Keras-Tuner search is completed,
    retrieves the different trials hyperparameters values
    from the respective json dumps.
    """
    trial_folders = [f for f in os.listdir(tuner_project_dir)
                     if os.path.isdir(os.path.join(tuner_project_dir, f)) and re.match(r'trial.*', f)]
    trials_hp = pd.DataFrame()
    for trial_folder in trial_folders :
        # REMARK : As per the "os.listdir" documentation,
        # the list of events files is in arbitrary order !
        # (thus we retrieve the 'creation_time' folder property
        #  for chronological sorting purpose)
        folder_creation_time = os.stat(os.path.join(tuner_project_dir, trial_folder)).st_ctime
        #print(folder_creation_time)

        with open(os.path.join(tuner_project_dir, trial_folder, 'trial.json')) as trial_file:
            trial_dict = json.load(trial_file)
            #display(trial_dict)
        trial_hp = \
            pd.concat([pd.DataFrame({'trial_id' : trial_dict['trial_id']}
                                    , index=[0])
                       , pd.DataFrame.from_records(
                           trial_dict['hyperparameters']['values'], index=[0])
                      ], axis=1)
        trial_hp['val_loss (best)'] = \
            trial_dict['metrics']['metrics']['val_loss']['observations'][0]['value']
        trial_hp['folder_creation_time'] = folder_creation_time
        trials_hp = pd.concat([trials_hp, trial_hp])
    trials_hp = trials_hp.sort_values(by='folder_creation_time', ascending=True) \
                    .drop(['folder_creation_time'], axis=1).reset_index(drop=True)
    trials_hp.index.names = ['trial']

    return trials_hp


#/////////////////////////////////////////////////////////////////////////////////////






















