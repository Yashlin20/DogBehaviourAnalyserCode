class LogEntryGrouped(object):

    def __init__(self,action, start_time, end_time):
        self.startTime = start_time
        self.endTime = end_time
        self.action = action

    def print_TimeStamp(self):

        start_time = self.startTime.strftime("%H:%M:%S")
        end_time = self.endTime.strftime("%H:%M:%S")
        print(str(self.action) + '         ' + str(start_time) + ' - ' + str(end_time))

    def output_line(self):

        txt = "{:<30}{:>9} -{:>9} "


        start_time = self.startTime.strftime("%H:%M:%S")
        end_time = self.endTime.strftime("%H:%M:%S")
        return txt.format(str(self.action),str(start_time),str(end_time))

        #return str(self.action) + '         ' + str(start_time) + ' - ' + str(end_time)

    def get_end_time(self):
        return self.endTime