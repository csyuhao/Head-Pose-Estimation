'''
    before using mutiprocessing detect OS
'''

from platform import system

def detect_os(bypass = False):
    ''' check OS, as mutiprocessing may not work properly on Windows and macOS '''
    if bypass is True:
        return
    
    os_name = system()

    if os_name in ['Windows']:
        print("It seems that you are running this code from {}, on which the Python multiprocessing may not work properly. Consider running this code on Linux.".format(os_name))
    else:
        print("Linux is fine! Python multiprocessing works.")