
import logging
class LoggerImpl:
    def GetLogger(self):
        logLevel = logging.INFO
        if not hasattr(self, "logger") or self.logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(logLevel)
            if not self.logger.handlers:
                formatter = logging.Formatter(f'>>> %(asctime)s - %(levelname)s - [{self.__class__.__name__}] %(message)s')
                ch = logging.StreamHandler()
                ch.setLevel(max(logLevel, logging.INFO))
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)
                fh = logging.FileHandler(f"{self.__class__.__name__}.log")
                fh.setLevel(logLevel)
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)
        return self.logger

    def Debug(self, msg):
        self.GetLogger().debug(msg)

    def Info(self, msg):
        self.GetLogger().info(msg)

    def Warning(self, msg):
        self.GetLogger().warning(msg)

    def Error(self, msg):
        self.GetLogger().error(msg)
