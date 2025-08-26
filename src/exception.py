# here we are create the custom exception handler okkay
import sys 
from src.logger import logging

def error_message_detial(error,error_detail:sys): 
    """ this function taks error and its message from the python interpreter and 
    return in the well formatted form message okkkay , her sys is use for the accessing the error and message
    the python interpreter and
    runtime environment 
    """
    _,_,exc_tb = error_detail.exc_info() # This return 3 parameter
    file_name = exc_tb.tb_frame.f_code.co_file_name # extracting the file name
    error_message = f"Error Occurred in Script : [{file_name}] at line number: [{exc_tb.tb_lineno}] with messsage [{str(error)}]"
    
    return error_message

class CustomException(Exception):
    """ Custom Exception class that inherited from the build in class Exception class. 
    It formats the error message using the error_message_detial function 
    """
    def __init__(self,error,error_detail:sys):
        super().__init__()
        self.error_message = error_message_detial(error,error_detail)
        # this constructor contains itself variables , which is error_message_detail
        
    def __str__(self): # This is str dunder method use for return teh error in string format
        return self.error_message     
      
# yeah that's it and it's the basic custom exception handler for my project okkay 
      