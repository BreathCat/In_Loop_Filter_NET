import sys
savedStdout = sys.stdout  # 保存标准输出流

class Logger(object):  # redirect std output
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger("aaattt.log", sys.stdout)
# sys.stderr = Logger(train.log_file, sys.stderr)		# redirect std err, if necessary

# now it works
print("main start")


sys.stdout = savedStdout  # 恢复标准输出流
print ('This message is for screen!')

