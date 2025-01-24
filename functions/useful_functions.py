

class color:
    PURPLE = '\u001b[0;35m'
    CYAN = '\u001b[36;1m'
    DARKCYAN = '\u001b[46m'
    BLUE = '\u001b[34m'
    GREEN = '\u001b[32m'
    YELLOW = '\u001b[33m'
    RED = '\u001b[31m'
    BOLD = '\u001b[4m'
    UNDERLINE = '\u001b[31;1;4m'
    END = '\u001b[0m'
    BLACK = '\u001b[30m'

def identify_character_variables(
#    self,
    input_data
    ): 
    
    return input_data.columns[input_data.dtypes == object]
    
def identify_numeric_variables(
#    self,
    input_data
    ): 
    
    return input_data.columns[input_data.dtypes != object]


