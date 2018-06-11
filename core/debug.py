# 3rd moddule
import logging

# native module
import os

def log(msg):
    '''Log format'''
    import logging
    logging.basicConfig(level=logging.DEBUG,
                        format=' %(asctime)s - %(levelname)s - %(message)s')
    if type(msg) == str:
        logging.disable(logging.CRITICAL)
        logging.debug(msg)
    else:
        logging.debug('logging parameter type wrong')

def msg(msg, num = 30):
    '''Print output format'''
    msg = str(msg)
    print(msg.ljust(num,'-'))
