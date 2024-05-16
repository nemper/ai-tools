import logging
import sys
import subprocess
import functools
import traceback

class LoggingUtility:
    def __init__(self, log_file_name="everything.log"):
        self.logger = logging.getLogger(__name__)
        self.setup_logging(log_file_name)

    def setup_logging(self, log_file_name):
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_name),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.captureWarnings(True)

    def log_all_calls(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            self.logger.debug(f"Calling {func.__name__}({signature})")

            try:
                value = func(*args, **kwargs)
                self.logger.debug(f"{func.__name__} returned {value!r}")
                return value
            except Exception as e:
                error_msg = f"Exception in {func.__name__}: {str(e)}"
                self.logger.error(error_msg)
                self.logger.debug(traceback.format_exc())
                raise
        return wrapper

    def capture_system_logs(self):
        try:
            if sys.platform.startswith('win'):
                command = ['wevtutil', 'qe', 'Application', '/c:10', '/rd:true', '/f:text']
            else:
                command = ['journalctl', '-n', '10', '--no-pager']
            logs = subprocess.check_output(command, text=True)
            self.logger.info("System logs captured successfully")
            self.logger.debug(logs)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to capture system logs: {e}")




# Instantiate the logging utility
log_util = LoggingUtility()

# Example function to be logged
@log_util.log_all_calls
def test_function(param1, param2):
    return param1 + param2

# Using the class to capture system logs
if __name__ == "__main__":
    print(test_function(5, 7))
    log_util.capture_system_logs()