import logging
import logging.handlers
import time

class Logger():
    def __init__(self, log_file):
        self.log_file = '../log/' + log_file + '.log'
        self.print_log('log init.')

    def logger_print(self, msg):
        logger = logging.getLogger(self.log_file)
        if not logger.handlers:        
            logger.setLevel(logging.INFO)
            rh=logging.handlers.TimedRotatingFileHandler(self.log_file,'D')
            fm=logging.Formatter("[%(asctime)s] %(message)s","%Y-%m-%d %H:%M:%S")
            rh.setFormatter(fm)
            logger.addHandler(rh) 
        logger.info(msg)

    def print_log(self, msg):
        self.logger_print(msg)
        print(msg)
        
    def print_dict(self, dic, order = None):
        key_list = list(dic.keys())
        if(order) :
            li = [i for i in key_list if i not in order] 
            key_list = order + li
        for key in key_list:
            if not dic.__contains__(key):
                continue
            if(isinstance(dic[key], dict)) :
                self.print_log(('%-16s') % key + ': {')
                for key2 in dic[key].keys():
                    self.print_log(('%20s') % key2 + ': ' + str(dic[key][key2]))
                self.print_log('%-19s' % '}')
            else:
                self.print_log(('%-16s') % key + ': ' + str(dic[key]))
            
                     
        