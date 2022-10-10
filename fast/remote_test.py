import fast.remote_file_save as remote_file_save
import time

time_wait = 20
while True:
    print(remote_file_save.saveData())
    time.sleep(time_wait)