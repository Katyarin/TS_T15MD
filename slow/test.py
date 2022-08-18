import Connect_slow
import time

time_wait = 5
shotn  = 1
while True:
    print(Connect_slow.getValue(shotn))
    shotn+=1
    time.sleep(60*time_wait)
