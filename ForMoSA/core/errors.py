class ForMoSAError(Exception):
    """Main ForMoSA error class."""

    def __init__(self, msg, logger=None):
        super().__init__(msg)
        if logger:
            logger.error(msg)
