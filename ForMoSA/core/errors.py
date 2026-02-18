# detinition of main ForMoSA error class
class ForMoSAError(Exception):
    def __init__(self, msg, logger=None):
        super().__init__(msg)
        if logger:
            logger.error(msg)
