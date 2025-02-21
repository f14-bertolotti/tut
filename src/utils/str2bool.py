def str2bool(value:str):
    match value:
        case "True"  : return True
        case "False" : return False
        case "true"  : return True
        case "false" : return False
        case "1"     : return True
        case "0"     : return False
        case _: raise ValueError(f"Cannot convert {value} to bool")
 
