import os
import datetime


class Logger(object):
    def __init__(self, path=".", name="deployment.log"):
        self._path = path
        self._name = name
        self._log_file = os.path.join(path, name)

    def log(self, level, string):
        with open(self._log_file, 'a') as f:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = "{} - {} - {}".format(now, level, string)
            print(message)
            message += '\n'
            f.writelines(message)

    def info(self, message):
        self.log('INFO', message)

    def warn(self, message):
        self.log('WARNING', message)

    def error(self, message):
        self.log('ERROR', message)


if __name__ == "__main__":
    log = Logger()
    log.info("123")