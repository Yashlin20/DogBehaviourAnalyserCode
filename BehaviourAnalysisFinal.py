import warnings
warnings.filterwarnings("ignore")
from LogEntryGrouped import LogEntryGrouped
from LogEntry import LogEntry
from datetime import timedelta
import os
import cv2
from datetime import datetime
import keras
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

image_size = 224
classes = ['Eating or Sniffing', 'Laying Down', 'Sitting', 'Standing', 'Walking']


# Method to determine if the behaviour is an active behaviour or not
def is_active_behaviour(behaviour):
    if (behaviour == 'Eating or Sniffing' or behaviour == 'Standing' or behaviour == 'Walking'):
        return True
    else:
        return False

# Method used to display each frame. Each frame is displayed and captioned with the
# current time as well as the predicted behaviour
def display_frame(frame, prediction, current_time):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame, (0, 0), (1200, 100), (0, 0, 0), -1)
    cv2.putText(frame,
                current_time + '  ' + prediction,
                (50, 50),
                font, 2,
                (255, 128, 0),
                2,
                cv2.LINE_4)
    cv2.imshow('Frame', cv2.resize(frame, (800, 800)))



# Method to increment the amount of active time of the dog. This active time is used to alert the owner to the dog being unsettled
def active_behaviour_alert(prediction, previous_prediction, current_time, previous_time, active_time_total):
    if(is_active_behaviour(prediction) == True and is_active_behaviour(previous_prediction) == True):
        return (active_time_total + (current_time - previous_time))
    else:
        return active_time_total


def to_datetime_object(date_string, date_format):
    s = datetime.strptime(date_string, date_format)
    return s


# Method used to extract the exact behaviour the model predicts for each frame
def get_prediction(predictionMatrix):

    predictions = predictionMatrix[0]
    high_index = 0
    high_value = predictions[0]
    count = 0
    for prediction in predictions:
        if(prediction > high_value):
            high_value = prediction
            high_index = count
        count = count + 1
    return classes[high_index]

# Method used to group individual predictions into a group of behaviours
def group_actions(log_entry_array):
    grouped_log_entry_array = []
    i = 0
    start_time = log_entry_array[i].time
    while(i < len(log_entry_array) - 1):
        current_action = log_entry_array[i].action
        start_time = log_entry_array[i].time
        end_time = start_time
        j = i + 1
        while(log_entry_array[j].action == current_action and j < len(log_entry_array) - 2):
            end_time = log_entry_array[j].time
            j = j + 1
        grouped_log_entry_array.append(LogEntryGrouped(current_action, start_time, end_time))
        i = j
    return grouped_log_entry_array


# Method used to regroup an already existing group of behaviours after the misclassified behaviours have been renamed
# into their correct behaviours
def group_actions2(grouped_log_entry_array):
    grouped_log_entry_array_regrouped = []
    i = 0
    while(i < len(grouped_log_entry_array) - 1):
        current_action = grouped_log_entry_array[i].action
        start_time = grouped_log_entry_array[i].startTime
        end_time = grouped_log_entry_array[i].endTime
        j = i + 1
        while(grouped_log_entry_array[j].action == current_action and j < len(grouped_log_entry_array) - 2):
            end_time = grouped_log_entry_array[j].endTime
            j = j + 1
        grouped_log_entry_array_regrouped.append(LogEntryGrouped(current_action, start_time, end_time))

        i = j
    return grouped_log_entry_array_regrouped

# Method used to iterate through a behaviours that have been grouped and remove the behaviours that were misclassified
# due to the transition of the dog from behaviour to behaviour
def remove_misclassified(grouped_log_entry_array):
    grouped_array_with_no_misclassifications = grouped_log_entry_array
    i = 1
    while(i < len(grouped_log_entry_array) - 2):
        currentAction = grouped_log_entry_array[i].action
        time_difference = (grouped_log_entry_array[i].endTime) - (grouped_log_entry_array[i].startTime)
        td_secs = int(round(time_difference.total_seconds()))
        if(td_secs == 0 and ((grouped_log_entry_array[i - 1].action) == (grouped_log_entry_array[i + 1].action))) :
            grouped_array_with_no_misclassifications[i].action = grouped_log_entry_array[i - 1].action
        i = i + 1
    return(group_actions2(grouped_array_with_no_misclassifications))

# Method used to return the total amount of time a dog was active for as well as the total amount of time a dog had performed
# particular behaviours for during the system's execution. The start time and end time were also captured and returned here
def get_total_active_time(arr):
    time = timedelta(0)
    time_eating_sniffing = timedelta(0)
    time_standing = timedelta(0)
    time_walking = timedelta(0)
    time_laying_down = timedelta(0)
    time_sitting = timedelta(0)
    end_time = datetime(2021,11,16)
    start_time = datetime(2021,11,16)
    i = 0
    for log in arr:
        if(i == 0):
            start_time = log.startTime
        if(log.action == 'Eating or Sniffing'):
            action_time = log.endTime - log.startTime
            time = time + action_time
            time_eating_sniffing = time_eating_sniffing + action_time
        elif(log.action == 'Standing'):
            action_time = log.endTime - log.startTime
            time = time + action_time
            time_standing = time_standing + action_time
        elif(log.action == 'Walking'):
            action_time = log.endTime - log.startTime
            time = time + action_time
            time_walking = time_walking + action_time
        elif(log.action == 'Laying Down'):
            non_action_time = log.endTime - log.startTime
            time_laying_down = time_laying_down + non_action_time
        else:
            non_action_time = log.endTime - log.startTime
            time_sitting = time_sitting + non_action_time
        i = i + 1
        end_time = log.endTime

    return time, time_eating_sniffing, time_standing, time_walking, time_laying_down, time_sitting, end_time, start_time


# Used to capture the user's email address, dog's name and
# video feed to be monitored (would usually be camera input if the system was used in real world)
video_name = input('Enter the name of the video feed to be monitored : ')
email_address = input('Please enter your email : ')
dog_name = input('Enter the name of the dog : ')



# Connecting to the smtp server on port 465
mail = smtplib.SMTP_SSL('smtp.gmail.com', 465)


try:

    # creating a folder named LogFiles
    if not os.path.exists("LogFiles/" + str(dog_name)):
        os.makedirs("LogFiles/" + str(dog_name))

# if not created then raise error
except OSError:
    print('Error: Creating directory of data')



# Read the video from specified path
cam = cv2.VideoCapture(video_name + ".mp4")

# Used to keep track of the current frame of the video feed
current_frame = 0

# Model loaded into system.
model = keras.models.load_model('model21-epoch40-batch1-B0AdamNoPP-DO=0.2UpdatedDataset7525')
# Array used to store actions of dog throughout the execution of system
prediction_time_stamps = []
# Array used to store actions of dog after every unsettling behaviour check. This array records the actions to be
# emailed in the case of unsettling behaviour
mini_prediction_time_stamps = []

# Variables used throughout the program's execution
total_time = timedelta(0)
active_time_net = timedelta(0)
previous_behaviour = ''
previous_time = datetime.now()
first_email_flag = False
email_times = []
upperbound_time = timedelta(minutes = 1)
five_minute_interval_count = 0
unsettled_behaviour_five_minute_interval_count = 0


while (True):

    # Reading from frame
    ret, frame = cam.read()

    # If video is still left continue creating images
    if ret :

        # Reading every 5 frames to ensure program runs in correct time. Anything less than 3 causes system to run slowly.
        if(current_frame % 3 == 0):

            # Converting image to RGB since model is trained on RGB images
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Scaling image to the size in which the model was trained
            image = cv2.resize(image, (image_size, image_size))
            # Reshaping the image to the shape in which the model was trained on.
            image = image.reshape(1, image_size, image_size, 3)

            # Using model to predict the behaviour of the dog in the image
            prediction = model.predict(image)

            # Getting the current time
            now = datetime.now()

            # Getting the current time, to be displayed on video feed
            current_time = now.strftime("%H:%M:%S")

            # Storing behaviour and corresponding time in a LogEntry object
            behaviour = LogEntry(str(get_prediction(prediction)), now)

            prediction_time_stamps.append(behaviour)
            mini_prediction_time_stamps.append(behaviour)
            total_time = total_time + (now - previous_time)
            active_time_net = active_behaviour_alert(str(get_prediction(prediction)), previous_behaviour, now, previous_time, active_time_net)


            # Check done every 30 minutes to look for unsettling behaviour
            if (five_minute_interval_count == 6): #for test return values
                five_minute_interval_count = 0

                # Check to see if the dog was active for less than 20 minutes, if it was active for less than 20 minutes,
                # an email is sent to the owner informing them that the dog is no longer unsettled and the system stops
                # evaluating the dog every 5 minutes, but rather every 30 minutes again
                if(unsettled_behaviour_five_minute_interval_count < 4): #for test return values
                    first_email_flag = False

                    body = str(dog_name) + " is no longer unsettled, his behaviour is now normal."
                    subject = str(dog_name) + " No Longer Exhibiting Unusual Behaviour"
                    message = MIMEMultipart()
                    message['From'] = 'behaviouranalysis22@gmail.com'
                    message['To'] = email_address
                    message['Subject'] = subject
                    message.attach(MIMEText(body, 'plain'))
                    text = message.as_string()
                    mail.sendmail("behaviouranalysis22@gmail.com", email_address, text)
                unsettled_behaviour_five_minute_interval_count = 0


            # Method to check if first email was sent and then if so, a further check is done every 5 minutes to notify
            # owner to continuation of unusual behaviour
            if (first_email_flag == True and (
                    (total_time // 1000000 * 1000000) % timedelta(seconds = 30) == timedelta(seconds=0)) and (
                    (total_time // 1000000 * 1000000) not in email_times) and ((total_time//1000000*1000000 == timedelta(0)) == False)):
                threshold_value_email = (((total_time) * 3 / 4) // 1000000 * 1000000) - timedelta(seconds = 15)
                # Check done to see if the dog was still active for threshold value calculated as 3/4 of 5 minutes
                if (active_time_net > threshold_value_email):

                    body = str(dog_name) + " is still unsettled" + "\n" + "These are the behaviours he/she has exhibited :" + "\n"
                    subject = str(dog_name) + " Still Exhibiting Unusual Behaviour"
                    message = MIMEMultipart()
                    message['From'] = 'behaviouranalysis22@gmail.com'
                    message['To'] = email_address
                    message['Subject'] = subject
                    body = body + "====================================================================================\n"

                    temp = group_actions(mini_prediction_time_stamps)
                    temp2 = remove_misclassified(temp)
                    # Adding the log of the dog's unsettling activities to the body of the email
                    for x in temp2:
                        body = body + (str(x.output_line()) + "\n")

                    body = body + "====================================================================================\n"
                    body = body + "The dog has been monitored for " + str(total_time//1000000*1000000) + " from which he/she has been active for " + str(active_time_net//1000000*1000000) + "\n"
                    body = body + "The dog should be relaxed for a time period less than " + str(threshold_value_email) + " according to the time he/she was monitored for"

                    message.attach(MIMEText(body, 'plain'))
                    text = message.as_string()
                    mail.sendmail("behaviouranalysis22@gmail.com", email_address, text)
                    total_time = timedelta(0)
                    active_time_net = timedelta(0)
                    mini_prediction_time_stamps = []
                    # Incrementing the number of 5 minute intervals the dog was unsettled for
                    unsettled_behaviour_five_minute_interval_count = unsettled_behaviour_five_minute_interval_count + 1
                    # Incrementing the number of 5 minute intervals checked
                    five_minute_interval_count = five_minute_interval_count + 1
                else:
                    total_time = timedelta(0)
                    active_time_net = timedelta(0)
                    mini_prediction_time_stamps = []
                    five_minute_interval_count = five_minute_interval_count + 1


            # Method used to send first email after unusual activity had occurred during the 30 minute period
            if(total_time//1000000*1000000 == upperbound_time and first_email_flag == False):
                threshold_value_email = (((total_time) * 3 / 4) // 1000000 * 1000000)
                if(active_time_net > threshold_value_email):
                    print('Unusual behaviour')

                    mail.login("behaviouranalysis22@gmail.com", "fourarm$")
                    body = str(dog_name) + " is unsettled" + "\n" + "These are the behaviours he/she has exhibited :" + "\n"
                    subject = str(dog_name) + " Exhibiting Unusual Behaviour"
                    message = MIMEMultipart()
                    message['From'] = 'behaviouranalysis22@gmail.com'
                    message['To'] = email_address
                    message['Subject'] = subject
                    body = body + "====================================================================================\n"

                    temp = group_actions(mini_prediction_time_stamps)
                    temp2 = remove_misclassified(temp)
                    for x in temp2:
                        body = body + (str(x.output_line()) + "\n")


                    body = body + "====================================================================================\n"
                    body = body + "The dog has been monitored for " + str(total_time//1000000*1000000) + " from which he/she has been active for " + str(active_time_net//1000000*1000000) + "\n"
                    body = body + "The dog should be relaxed for a time period less than " + str(threshold_value_email) + " according to the time he/she was monitored for"

                    message.attach(MIMEText(body, 'plain'))
                    text = message.as_string()
                    mail.sendmail("behaviouranalysis22@gmail.com", email_address, text)
                    total_time = timedelta(0)
                    active_time_net = timedelta(0)
                    mini_prediction_time_stamps = []
                    first_email_flag = True
                else:
                    total_time = timedelta(0)
                    active_time_net = timedelta(0)
                    mini_prediction_time_stamps = []

            previous_time = now
            previous_behaviour = str(get_prediction(prediction))


            display_frame(frame, str(get_prediction(prediction)), str(current_time))

            # 'q' Key used to stop stream if it is not needed, the system will monitor the dog in the background
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        current_frame += 1
    else:
        break


final_array = group_actions(prediction_time_stamps)

final_array_no_misclassified = remove_misclassified(final_array)

times = get_total_active_time(final_array_no_misclassified)

total_active_time = times[0]


file1 = open("LogFiles/" + str(dog_name) + "/" + now.strftime("%Y%m%d") + ".txt","w")
file1.write("Action Log Report for : " + str(dog_name) + "\n")
file1.write("Date : " + now.strftime("%Y/%m/%d") + "\n")
file1.write("Start Time : " + str(times[7].strftime("%H:%M:%S")) + "\n")
file1.write("End Time : " + str(times[6].strftime("%H:%M:%S")) + "\n")
file1.write("Total Time : " + str((times[6]-times[7])//1000000*1000000) + '\n')
file1.write("\n")
file1.write("================================================================================  \n")


for m in final_array_no_misclassified:
    file1.write(str(m.output_line()) + "\n")
file1.write('================================================================================  \n')

file1.write('Total Time Spent Eating or Sniffing : ' + str(times[1]//1000000*1000000) + '\n')
file1.write('Total Time Spent Standing : ' + str(times[2]//1000000*1000000) + '\n')
file1.write('Total Time Spent Walking : ' + str(times[3]//1000000*1000000) + '\n')
file1.write('Total Time Spent Laying Down : ' + str(times[4]//1000000*1000000) + '\n')
file1.write('Total Time Spent Sitting : ' + str(times[5]//1000000*1000000) + '\n')
file1.write('Total Active Time : ' + str(total_active_time//1000000*1000000) + '\n\n\n')
# Release all space and windows once done
mail.quit()
cam.release()
cv2.destroyAllWindows()
