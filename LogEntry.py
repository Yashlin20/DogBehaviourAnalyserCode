class LogEntry:

    def __init__(self,action, time):
        self.time = time
        self.action = action

    def print_TimeStamp(self):


        current_time = self.time.strftime("%H:%M:%S")

        print(str(self.action) + '         ' + str(current_time))