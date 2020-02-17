'''
Created on Feb 14, 2020

@author: USER
'''
from train import train
import gc
from parameters import  date_now,LOSS,CELL,N_STEPS,N_LAYERS,UNITS,ticker, LOOKUP_STEP
import tensorflow as tf
tf.test.gpu_device_name()
for step in range(1,LOOKUP_STEP):
    model_name = "{now}_{ticker_name}-{error_loss}-{cell_name}-seq-{sequence_lenght}-step-{step}-layers-{layers}-units-{neurons}".format(
        now=date_now,
        ticker_name=ticker,
        error_loss=LOSS,
        cell_name=CELL.__name__,
        sequence_lenght=N_STEPS,
        step=step,
        layers=N_LAYERS,
        neurons=UNITS
    )
    gc.collect()
    train(step, model_name)
    gc.collect()
    