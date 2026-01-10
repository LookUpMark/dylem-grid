import os
import logging
import warnings
import optuna

def suppress_logs():
    """
    Suppress verbose logs and warnings from PyTorch Lightning, Optuna, and TensorFlow.
    Should be called at the start of notebooks/scripts.
    """
    # 1. Filter Warnings
    warnings.simplefilter('ignore')
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', '.*Tip:.*')
    warnings.filterwarnings('ignore', '.*GPU available.*')

    # 2. Suppress specific rank_zero loggers (GPU info, version tips)
    logging.getLogger('lightning.pytorch.utilities.rank_zero').setLevel(logging.WARNING)
    logging.getLogger('pytorch_lightning.utilities.rank_zero').setLevel(logging.WARNING)
    logging.getLogger('pytorch_lightning.accelerators.cuda').setLevel(logging.WARNING)
    
    # Also suppress main loggers just in case
    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
    logging.getLogger('lightning').setLevel(logging.ERROR)

    # 3. Environment Variables
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['SLURM_JOB_NAME'] = 'bash'

    # 4. Optuna Silence
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)
