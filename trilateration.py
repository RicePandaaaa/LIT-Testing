import os
import time

os.system("cd ./home/axis")        #default to home directory to run command
while(True):
  os.system("iwconfig wl2s0 | grep 'Signal level'")      #grab and print RSSI values in dBm
  time.sleep(0.5)                                        #delay so user can read the values, otherwise the system prints too quickly
