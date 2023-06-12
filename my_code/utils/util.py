import datetime

def get_time():
    now = datetime.datetime.now()
    eastern_eight = now+datetime.timedelta(hours=8)
    return eastern_eight.strftime('%Y-%m-%d_%H:%M:%S')