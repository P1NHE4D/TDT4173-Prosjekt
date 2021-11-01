import geojson
import numpy as np
import pandas as pd


def geojson_to_csv(path):
    with open(path) as f:
        gj = geojson.load(f)
    features = gj['features']
    coordinates = []

    for feature in features:
        c = feature["geometry"]["coordinates"]
        coordinate_type = feature["geometry"]["type"]
        if coordinate_type == "Polygon":
            sum_building_coordinates = np.array([0.0, 0.0])
            for building in c:
                building_coordinates = np.array(building).mean(axis=0)
                sum_building_coordinates += building_coordinates
            coordinate = (sum_building_coordinates / len(c)).round(6)
        elif coordinate_type == "Point":
            coordinate = np.array(c).round(6)
        elif coordinate_type == "LineString":
            coordinate = np.array(c).mean(axis=0).round(6)
        elif coordinate_type == "MultiPolygon":
            c = c[0]
            sum_building_coordinates = np.array([0.0, 0.0])
            for building in c:
                building_coordinates = np.array(building).mean(axis=0)
                sum_building_coordinates += building_coordinates
            coordinate = (sum_building_coordinates / len(c)).round(6)
        else:
            raise Exception("Unknown coordinate type: {}".format(coordinate_type))
        coordinates.append(coordinate)
    df = pd.DataFrame(coordinates, columns=["longitude", "latitude"])
    df = df.reindex(columns=["latitude", "longitude"])
    return df


if __name__ == '__main__':
    df = geojson_to_csv("../data/hospitals.geojson")
    df.to_csv("hospitals.csv")
    df = geojson_to_csv("../data/schools.geojson")
    df.to_csv("schools.csv")
