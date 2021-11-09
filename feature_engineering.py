import geojson
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import geopy.distance
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def one_hot_encode(data, categories, orig_var, remove_org=False):
    for k, category in categories.items():
        data[category] = (data[orig_var] == k).astype(int)
    if remove_org:
        data.drop(orig_var, axis=1, inplace=True)
    return data


def distance_to_center_feature(data):
    moscow_center = [55.751244, 37.618423]
    coordinates = data[['latitude', 'longitude']].to_numpy()
    dist = [geopy.distance.distance(moscow_center, coordinate).km for coordinate in coordinates]
    data['distance_to_center'] = dist


def bathrooms_feature(data):
    data["bathrooms"] = data.bathrooms_shared + data.bathrooms_private


def has_seller_feature(data):
    for ind, row in data.iterrows():
        data.at[ind, "has_seller"] = pd.isna(row.seller)
    data['has_seller'] = data['has_seller'].astype(int)


def distance_to_metro_feature(data, add_metro_lines=True):
    d = data.copy()

    metro_data = pd.read_csv("data/moscow_metro_data.csv", delimiter=";")

    # drop duplicates
    ind = metro_data[metro_data["English transcription"].duplicated()].index
    metro_data = metro_data.drop(ind)
    metro_data = metro_data.reset_index()
    metro_lines = ["metro_line_{}".format(i) for i in range(1, 16)]

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(metro_data[["latitude", "longitude"]])
    distances, indices = nbrs.kneighbors(d[["latitude", "longitude"]])
    d["metro_index"] = indices

    for ind, row in d.iterrows():
        metro_coordinate = metro_data.loc[[row.metro_index]][["latitude", "longitude"]].to_numpy()
        distance = geopy.distance.distance([[row["latitude"], row["longitude"]]], metro_coordinate).km
        data.at[ind, "distance_to_metro"] = distance

        if add_metro_lines:
            number_of_metro_lines = 0
            metro_lines_data = metro_data.loc[[row.metro_index]][metro_lines]
            for metro_line in metro_lines:
                access_to_line = metro_lines_data[metro_line].values[0]
                data.at[ind, metro_line] = access_to_line
                if access_to_line == 1:
                    number_of_metro_lines += 1
            data.at[ind, "number_of_metro_lines"] = number_of_metro_lines


def bearing_feature(data):
    for ind, row in data.iterrows():
        moscow_center = [55.751244, 37.618423]
        c1 = moscow_center
        c2 = [row["latitude"], row["longitude"]]
        y = np.sin(c2[1] - c1[1]) * np.cos(c2[0])
        x = np.cos(c1[0]) * np.sin(c2[0]) - np.sin(c1[0]) * np.cos(c2[0]) * np.cos(c2[1] - c1[1])
        rad = np.arctan2(y, x)
        bearing = ((rad * 180 / np.pi) + 360) % 360
        data.at[ind, "bearing"] = bearing


def distance_to_hospital_feature(data):
    d = data.copy()
    hospital_data = pd.read_csv("data/hospitals.csv")
    hospital_data = hospital_data[["latitude", "longitude"]]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(hospital_data)
    distances, indices = nbrs.kneighbors(data[["latitude", "longitude"]])
    d["hospital_index"] = indices
    for ind, row in d.iterrows():
        apartment_coordinates = [[row["latitude"], row["longitude"]]]
        hospital_coordinate = hospital_data.loc[[row.hospital_index]].to_numpy()
        distance = geopy.distance.distance(apartment_coordinates, hospital_coordinate).km
        data.at[ind, "distance_to_hospital"] = distance


def elevator_feature(data):
    for ind, row in data.iterrows():
        data.at[ind, "elevator"] = row.elevator_passenger == 1 or row.elevator_service == 1


def room_size_avg_feature(data):
    data["room_size_avg"] = data.area_total / data.rooms


def area_total_bins_feature(data):
    data["area_total_0_50"] = (data.area_total <= 50).astype(int)
    data["area_total_51_100"] = ((data.area_total > 50) & (data.area_total <= 100)).astype(int)
    data["area_total_101_200"] = ((data.area_total > 100) & (data.area_total <= 200)).astype(int)
    data["area_total_201_300"] = ((data.area_total > 200) & (data.area_total <= 300)).astype(int)
    data["are_total_301_inf"] = (data.area_total > 300).astype(int)


def distance_to_airport_feature(data):
    d = data.copy()
    airport_data = [[55.5822, 37.2453], [55.2431, 37.5422], [55.3546, 37.1603], [55.3312, 38.96], [55.3042, 37.3026]]
    airport_data = pd.DataFrame(airport_data, columns=["latitude", "longitude"])

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(airport_data[["latitude", "longitude"]])
    distances, indices = nbrs.kneighbors(d[["latitude", "longitude"]])
    d["airport_index"] = indices

    for ind, row in d.iterrows():
        airport_coordinate = airport_data.loc[[row.airport_index]][["latitude", "longitude"]].to_numpy()
        distance = geopy.distance.distance([row["latitude"], row["longitude"]], airport_coordinate).km
        data.at[ind, "distance_to_airport"] = distance


def area_total_log_feature(data):
    data["area_total_log"] = np.log1p(data.area_total)


def remaining_area_feature(data):
    data["remaining_area"] = data.area_total - data.area_living - data.area_kitchen


def distance_to_state_university_feature(data):
    uni_coordinates = [55.70444300116007, 37.528611852796914]
    data["distance_to_state_uni"] = [geopy.distance.distance([row["latitude"], row["longitude"]], uni_coordinates).km
                                     for ind, row in data.iterrows()]


def distance_to_tech_uni_feature(data):
    uni_coordinates = [55.76666597872545, 37.68511242319504]
    data["distance_to_tech_uni"] = [geopy.distance.distance([row["latitude"], row["longitude"]], uni_coordinates).km for
                                    ind, row in data.iterrows()]


def wealthy_districts_features(data, config):
    with open("data/geojson/khamovniki_polygon_data.geojson") as f:
        gj_khamovniki = geojson.load(f)
    with open("data/geojson/yakimanka_polygon_data.geojson") as f:
        gj_yakimanka = geojson.load(f)
    with open("data/geojson/arbat_polygon_data.geojson") as f:
        gj_arbat = geojson.load(f)
    with open("data/geojson/presnensky_polygon_data.geojson") as f:
        gj_presnensky = geojson.load(f)
    with open("data/geojson/Tverskoy_polygon_data.geojson") as f:
        gj_tverskoy = geojson.load(f)
    khamovniki_polygon = Polygon(gj_khamovniki["geometries"][0]["coordinates"][0][0])
    yakimanka_polygon = Polygon(gj_yakimanka["geometries"][0]["coordinates"][0][0])
    arbat_polygon = Polygon(gj_arbat["geometries"][0]["coordinates"][0][0])
    presnensky_polygon = Polygon(gj_presnensky["geometries"][0]["coordinates"][0][0])
    tverskoy_polygon = Polygon(gj_tverskoy["geometries"][0]["coordinates"][0][0])
    for ind, row in data.iterrows():
        point = Point([row["longitude"], row["latitude"]])
        if config["is_in_khamovniki"]:
            data.at[ind, "is_in_khamovniki"] = 1 if khamovniki_polygon.contains(point) else 0
        if config["is_in_yakimanka"]:
            data.at[ind, "is_in_yakimanka"] = 1 if yakimanka_polygon.contains(point) else 0
        if config["is_in_arbat"]:
            data.at[ind, "is_in_arbat"] = 1 if arbat_polygon.contains(point) else 0
        if config["is_in_presnensky"]:
            data.at[ind, "is_in_presnensky"] = 1 if presnensky_polygon.contains(point) else 0
        if config["is_in_tverskoy"]:
            data.at[ind, "is_in_tverskoy"] = 1 if tverskoy_polygon.contains(point) else 0
        if config["distance_to_ulitsa_ostozhenka"]:
            coordinates = [55.73936870697431, 37.595905186336914]
            data.at[ind, "distance_to_ulitsa_ostozhenka"] = geopy.distance.distance([row["latitude"], row["longitude"]], coordinates).km
            