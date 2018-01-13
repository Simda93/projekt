import numpy as np
import pandas as pd
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from pathlib import Path
import os
import random
import json
import time

geolocator = Nominatim()

def add_to_dict(d,k,v):
    val =d.get(k,[])
    if v not in val:
        val.append(v)
        d[k]=val
    return d
def increment_dict(d,k):
    val =d.get(k,0)+1
    d[k]=val
    return d


df = pd.read_csv("slovakia_parl_election_2016.csv")
data = df.values
header = list(df)
key =[i for i in range(5993)]
key.remove(5992)   #Borovce pri piestanoch, volebna ucast bola 0
key.remove(776)    #hlasy zo zahranicia
data=data[key,:]

obvodovVKraji={}
okresovVObvode={}
obciVOkrese={}
okrskovVObci={}
for i in range(data.shape[0]):
    row =data[i,:]
    kraj = row[0]
    obvod = row[2]
    okres = row[4]
    obec = row[6]
    key =""+str(kraj)
    add_to_dict(obvodovVKraji,key,obvod)
    key = key+"."+str(obvod)
    add_to_dict(okresovVObvode,key,okres)
    key = key+"."+str(okres)
    add_to_dict(obciVOkrese,key,obec)
    increment_dict(okrskovVObci,obec)

def do_geocode(address):
    try:
        return geolocator.geocode(address)
    except Exception:
        print("waiting for geopy")
        time.sleep(5)
        return do_geocode(address)

geoInfo = []
location = geolocator.geocode("")
minula_obec = ""
for i in range(data.shape[0]):
    row = data[i, :]
    kraj = row[0]
    obvod = row[2]
    okres = row[4]
    obec = row[6]
    key =""+str(kraj)
    data[i,3]=len(obvodovVKraji[key])
    key = key+"."+str(obvod)
    data[i,5]=len(okresovVObvode[key])
    key = key+"."+str(okres)
    data[i,7]=len(obciVOkrese[key])
    data[i,8] = okrskovVObci[obec]
    if(minula_obec != str(obec)):
        time.sleep(1.1)
        location = do_geocode(str(obec)+" , Slovakia")
        print(location.address)
    minula_obec=str(obec)
    data[i,18] =location.latitude
    data[i,19]=location.longitude
    data[i,20]=location.raw["importance"]
    geoInfo.append(location.raw)

np.savetxt("geolocation_gain.csv",geoInfo,fmt='%s')
header[3]="#_of_wards_in_region"
header[5]="#_of_districts_in_ward"
header[7]="#_of_municipalities_in_district"
header[8]="#_of_precincts_in_municipality"
header[18]="latitude"
header[19]="longtitude"
header[20]="location_importance"
list_of_keyes = [3,5,7,8,18,19,20,9,10,11,12,13,14,15,16,78,79,90,91,17]
data = data[:,list_of_keyes]    #78,79 a 90,91 su pocty,percenta hlasov smeru a lsns
new_header=[]

for i in list_of_keyes:
    new_header.append(header[i])
np.savetxt("unshuffled.csv",data,fmt='%s')
np.savetxt("header.csv",new_header,fmt='%s',newline=" ")
