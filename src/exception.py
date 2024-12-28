import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.logger import logging

def error_message_detail(error, error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    filename=exc_tb.tb_frame.f_code.co_filename
    
    error_message="Error occured in python script name [{0}] lineno [{1}] error message [{2}]".format(filename, exc_tb.tb_lineno, str(error))
    return error_message

class CustomException:
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message=error_message_detail(error=error_message, error_detail=error_detail)
        
    def __str__(self) -> str:
        return self.error_message

if __name__=='__main__':
    try:
        b=12/6
        print(b)
        a=10/0
        
    except Exception as e:
        print("cant divide with zero")
        raise CustomException(e,sys)