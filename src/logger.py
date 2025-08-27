# This is the logging python file okkay
import logging
import os
from datetime import datetime

# Create the log file with the specific format current time format (month,day,year:hour,minute,second)
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# Log path for the save this created file okkay or whenever logs are created i store this in this path
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)  # This specify the path not create
# here first para fetch the current directory (SUSPICIOUS TRANSACTION DETECTION) ,second create the folder , and third inside this logs folder create the logs file okkay

os.makedirs(logs_path,exist_ok=True) # here make directory means folder , with this countinues path only create the directory

LOG_FILE_PATH  = os.path.join(logs_path,LOG_FILE) # This the path of the .log file okkay 

# logging contains the basicConfigs 
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s -  %(message)s',
    level = logging.INFO,
    datefmt= '%Y-%m-%d %H:%M:%S'
)

# this the my basic logging file which is capable to save the logs of with the specific messages
