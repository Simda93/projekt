from geopy.geocoders import yandex
import numpy as np
geo = []
geolocator = yandex
location = geolocator.Yandex.geocode(self=geolocator,query="Michalovce, Slovakia")
print(location.address)
print((location.latitude, location.longitude))
print(location.raw)
