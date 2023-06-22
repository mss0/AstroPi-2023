import mysql.connector
import pandas as pd
import numpy as np
import math
import cv2
import os
import re
import csv

from collections import deque
from PIL import Image
from math import radians
from math import degrees

epsilon = 100
PI = 2 * math.acos(0)

# Access the database 
a = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "astropi",
    database = "astropi"
)


# Fetch all data from database to myresult
mycursor = a.cursor()
mycursor.execute("SELECT Latitude_departure, Longitude_departure, Latitude_arrival, Longitude_arrival, ID FROM airplanes_database")
myresult = mycursor.fetchall()

aux = [] 
vec = []

# We observed that the actual position of a photo at a given index x is consistently trailing the ISS by around 2000km
# This can be accounded for when checking the location of an image by using the coordinates of photo at index x - delay
delay = 23
INF = 1e9

for x in myresult:
  aux.append(x)


# Function to convert polar coordinates to cartesian ones
def conversion (angle):

  earth_radius = 6371
  angle = angle * PI / 180
    
  x = earth_radius * (math.cos(angle))
  y = earth_radius * (math.sin(angle))
    
  return x, y


# Function used to convert from cartesian to polar coordinates (radians)
def back_converter (*cartesian):
    
  return math.atan(cartesian[1] / cartesian[0])


for iterator in aux:

  x, y = conversion (iterator[0])
  x1, y1 = conversion (iterator[1])
  x2, y2 = conversion (iterator[2])
  x3, y3 = conversion(iterator[3])
    
  vec.append ((x, y, x2, y2, x1, y1, x3, y3, iterator[4] - 1))


# Read the excel file with the photo coordinates
excel = pd.read_excel('data.xls')


# Determine the intersections of the flights with the photo coordinates (at the exact time the photos were made) via the line equation
def line(*date):

  a = date[3] - date[1]
  b = date[0] - date[2]
  c = date[0] * (date[1] - date[3]) + date[1] * (date[2] - date[0])
    
  return a, b, c


# Check whether an intersection exists
def intersect (one, sec):

  a, b, c = line (sec[0], sec[1], sec[2], sec[3])
  a1, b1, c1 = line (sec[4], sec[5], sec[6], sec[7])
  yfunctie = (-a * one[0] - c) / b
  yfunctie1 = (-a1 * one[2] - c1) / b1
    
  # An epsilon approximation is neeeded as we could not obtain a bigger database
  if yfunctie + epsilon >= one[1] and yfunctie - epsilon <= one[1] and yfunctie1 + epsilon >= one[3] and yfunctie1 - epsilon <= one[3] :
    return 1, one[0], yfunctie, one[2], yfunctie1
  return 0, one[0], yfunctie, one[2], yfunctie1


data_dir = 'check'

photo_count = 661
photo_paths = ['placeholder'] * photo_count

p = []

# Load photos containing cirrus clouds and their coordinates
for directory in os.listdir(data_dir):

  curr_path = os.path.join(data_dir, directory)

  for photo in os.listdir(curr_path):

    index = int(re.findall(r'\d+', photo)[0])
    curr_time = excel.iloc[index - 1]['Time'].to_datetime64()
    curr_time = curr_time.astype(object)
    photo_paths[index - 1] = (os.path.join(curr_path, photo), curr_time)

    if directory == 'not cirrus':
      continue

    if index - delay < 1:
      continue
    
    p.append((excel.iloc[index - delay - 1]['Latitude'], excel.iloc[index - delay - 1]['Longitude'], index - 1))


new = []

for iterator in p:
    
  x, y = conversion(iterator[0])
  x1, y1 = conversion (iterator[1])
  new.append((x, y, x1, y1, iterator[2]))
  
avioane = [] # ((index of intersected photo), (tuple that contains the polar coordinates of the flight that intersected the photo)) 

# Check with all airplanes for an intersection
k = 0

for it in new:
    
  for first in vec:
      
    val, x, y, x1, y1 = intersect (it, first)
    lat = back_converter(x, y) * 180 / PI
    long = back_converter (x1, y1) * 180 / PI
      
    if val == 1:
      avioane.append((it[4], (lat, long, first[8], k)))
      k += 1

# Average speed of a plane is 900km/h
dist = 17 * 15 # (17 / 60 * 900)


# Find position of plane 17 minutes before it inteseced the photo
def solve (x, y, a, b, c):
    
  first = (a * a + b * b) / (b * b)
  aux = (c / b) + y
  sec = (2 * a * aux / b) - 2 * x
  last = x * x + aux * aux - dist * dist
  delta = sec * sec - 4 * first * last
    
  if delta >= 0:
      
    delta = math.sqrt(delta)
    xprim = (delta - sec) / (2 * first)
    yprim = (-a * xprim - c) / b
      
    return xprim, yprim
      
  else :
    return 0, 0
  
# Collect all positions of planes delayed by 17 minutes
avioane_back = []

for auto in avioane:
    
  x, y = conversion(auto[1][0])
  x1, y1 = conversion(auto[1][1])
  idx = auto[1][2]
  a, b, c = line (vec[idx][0], vec[idx][1], vec[idx][2], vec[idx][3])
  a1, b1, c1 = line (vec[idx][4], vec[idx][5], vec[idx][6], vec[idx][7])
  xback, yback = solve (x, y, a, b, c)
    
  if xback == 0 and yback == 0:
    avioane_back.append((-50.0034, -48.89))
    continue
      
  lat = back_converter (xback, yback) * 180 / PI
  xback1, yback1 = solve (x1, y1, a1, b1, c1)
    
  if xback1 == 0 and yback1 == 0:
    avioane_back.append((-50.0034, -48.89))
    continue
      
  long = back_converter (xback1, yback1) * 180 / PI
  avioane_back.append((lat, long))
    

# Make sure position is not out of the bounds of the image
def good(x, y): 
    
    width = 1296
    height = 972

    if x < 0 or x >= height:
        return 0
    if y < 0 or y >= width:
        return 0
    return 1


# Calculate light intensity for a given pixel
def formula_light_intensity(col, row, pix):

    r, g, b = pix[col, row]
    return math.sqrt( 0.299 * ( r ** 2 ) + 0.587 * ( g ** 2 ) + 0.114 * ( b ** 2 ) )


# Retrun average light inensity of an area on the photo
def lee(ys, xs, dist, mat, pix):
    
    sum = 0
    nr = 0
    
    dx = [0, 0, 1, -1]
    dy = [1, -1, 0, 0]
    
    q = deque()
    q.append((xs, ys))
    nr = 1
    sum = formula_light_intensity(ys, xs, pix)
    
    while len(q) != 0:
        
        x, y = q[0]
        q.popleft()
        
        if mat[x][y] == dist:
            continue
        
        for d in range(4):
            
            xn = x + dx[d]
            yn = y + dy[d]
            
            if good(xn, yn) and mat[xn][yn] == 0:
                
                mat[xn][yn] = mat[x][y] + 1
                nr = nr + 1
                sum = sum + formula_light_intensity(yn, xn, pix)
                q.append((xn, yn))
    
    return sum / nr


# Convert coordinates in the system with the origin in the center of the image to matrix indices
def find_pixel(img_path, dif_x, dif_y):

    img = Image.open(img_path)
    width, height = img.size
    ctr_col = width // 2
    ctr_row = height // 2
    dif_x = int(dif_x)
    dif_y = int(dif_y)

    pix_col = ctr_col + dif_x
    pix_row = ctr_row - dif_y

    return (pix_col, pix_row)


# Calculate light intensity for both the affected and the unaffected area
def zone_light_intensity(img_path, col_vechi, row_vechi, col_nou, row_nou):
    
    img = Image.open(img_path)
    width, height = img.size
    pixels = img.load()
    mat = [[0 for i in range(width)] for j in range(height)]
    
    dist = 5
    val_zona_veche = lee(col_vechi, row_vechi, dist, mat, pixels)
    mat = [[0 for i in range(width)] for j in range(height)]
    val_zona_noua = lee(col_nou, row_nou, dist, mat, pixels)
    
    return (val_zona_veche, val_zona_noua, val_zona_veche - val_zona_noua)


# To determine where the plane would appear in the photo we suppose that the observed surface on the Earth is planar
# This is a valid approximation as the angles at the center of the Earth that encompass said area are below 6 degrees


# Transpose coordinates from the plane of the Earth to the plane of the photo
# All Earth parameters are to be given in km and all photo parameters in px
def transform_to_photo(earth_a, earth_b, earth_x, earth_y, photo_a, photo_b):

    denom = earth_a**2 + earth_b**2 

    interm1, interm2 = (earth_x * earth_a + earth_y * earth_b) / denom, (earth_y * earth_a - earth_x * earth_b) / denom

    return photo_a * interm1 - photo_b * interm2, photo_b * interm1 + photo_a * interm2


# Assume the Earth is a sphere with the radius of 6371 km
earth_radius = 6371

# Get distance between two points on the Earth in km
def convert_to_km(lat1, long1, lat2, long2):

    return radians(long2 - long1) * earth_radius, radians(lat2 - lat1) * earth_radius


# Function to call the conversion
def photo_coordinates(lat1, long1, lat2, long2, plane_lat, plane_long, photo_x, photo_y):

    to_km1 = convert_to_km(lat1, long1, lat2, long2)
    to_km2 = convert_to_km(lat1, long1, plane_lat, plane_long)

    return transform_to_photo(to_km1[0], to_km1[1], to_km2[0], to_km2[1], photo_x, photo_y)
  

# Finds the part of the image that most resembles the template
def find_template(image, template):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    confidence = result.max()

    return np.where(result == confidence), confidence


# Calculate the position of the center of next photo relative to the center of the current one
# This done by juxtaposing the common parts of the images and is used to establish a coordinate system
def calculate_displacement(photo1, photo2): 

    (height, width) = photo1.shape[:2]
    (center_x, center_y) = (width // 2, height // 2)

    sections = [photo1[0:center_y, 0:center_x], photo1[0:center_y, center_x:width],
                photo1[center_y:height, 0:center_x], photo1[center_y:height, center_x:width]]

    top_left = [(0, 0), (center_x, 0), (0, center_y), (center_x, center_y)]

    best, best_index = 0, 0
    box = []

    for index, section in enumerate(sections):

        result = find_template(photo2, section)

        if result[1] < best:
            continue

        best, best_index = result[1], index
        box = result[0]

    return box[1][0] - top_left[best_index][0], box[0][0] - top_left[best_index][1]


# Use a csv file to store the results
header_list = ['plane_area', 'reference_area', 'diff']

def create_csv_file(data_file):
    
    with open(data_file, 'w') as f:
        
        writer = csv.writer(f)
        header = header_list
        writer.writerow(header)


def add_csv_data(data_file, data):

    with open(data_file, 'a') as f:
        
        writer = csv.writer(f)
        writer.writerow(data)

csv_path = 'data_file.csv'
csv_file = create_csv_file(csv_path)

# For each plane that intersects some photos, calculate the average impact is has on cloud luminosity
plane_count = len(aux) + 1

fq = np.zeros(plane_count)
count = np.zeros(plane_count)

for entry in avioane:

    photo_index = entry[0]

    if photo_index == 660:
      continue
    
    if photo_paths[photo_index + 1][1] - photo_paths[photo_index][1] > 11e9:
        continue
    
    if photo_index - delay < 0:
      continue

    if excel.iloc[photo_index - delay]['Latitude'] == excel.iloc[photo_index + 1 - delay]['Latitude'] and excel.iloc[photo_index - delay]['Longitude'] == excel.iloc[photo_index + 1 - delay]['Longitude']:
      continue

    displacement_x, displacement_y = calculate_displacement(cv2.imread(photo_paths[photo_index][0]),
                                                            cv2.imread(photo_paths[photo_index + 1][0]))

    plane_x, plane_y = photo_coordinates(excel.iloc[photo_index - delay]['Latitude'], excel.iloc[photo_index - delay]['Longitude'],
              excel.iloc[photo_index + 1 - delay]['Latitude'], excel.iloc[photo_index + 1 - delay]['Longitude'],
                                    entry[1][0], entry[1][1], displacement_x, displacement_y)
    
    idx = entry[1][3]
    
    prev_plane_x, prev_plane_y = photo_coordinates(excel.iloc[photo_index - delay]['Latitude'], excel.iloc[photo_index - delay]['Longitude'],
              excel.iloc[photo_index + 1 - delay]['Latitude'], excel.iloc[photo_index + 1 - delay]['Longitude'],
                                    avioane_back[idx][0], avioane_back[idx][1], displacement_x, displacement_y)

    pix_prev_col, pix_prev_row = find_pixel(photo_paths[photo_index][0], prev_plane_x, prev_plane_y)
    pix_plane_col, pix_plane_row = find_pixel(photo_paths[photo_index][0], plane_x, plane_y)

    if good(pix_prev_row, pix_prev_col) and good(pix_plane_row, pix_plane_col):
        
        light_prev, light_now, diff = zone_light_intensity(photo_paths[photo_index][0], pix_prev_col, pix_prev_row, pix_plane_col, pix_plane_row) 
        fq[entry[1][2]] +=  diff
        count[entry[1][2]] += 1

for x in range(plane_count - 1):
    
  if count[x] != 0:
      
    add_csv_data(csv_path, (x, fq[x] / count[x]))
