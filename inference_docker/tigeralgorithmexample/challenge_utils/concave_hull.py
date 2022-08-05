"""
Code from the tiger pathology baseline algorithm to find a non-convex hull
around the segmented tumour.

# Taken from:
# https://github.com/DIAGNijmegen/pathology-tiger-baseline/blob/main/concave_hull.py
"""

import os
import numpy as np
from shapely.ops import cascaded_union, polygonize
import shapely
import math
import shapely.geometry as geometry
import xml.etree.ElementTree as ET
import skimage.morphology
from scipy.spatial import Delaunay
import cv2
from ..rw import open_multiresolutionimage_image


def dist_to_px(dist, spacing):
    """ distance in um (or rather same unit as the spacing) """
    dist_px = int(round(dist / spacing))
    return dist_px


def mm2_to_px(mm2, spacing):
    return (mm2 * 1e6) / spacing ** 2


def alpha_shape(points, alpha):
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            return
        edges.add((i, j))
        edge_points.append([coords[i], coords[j]])

    coords = [(i[0], i[1]) if type(i) or tuple else i for i in points]
    tri = Delaunay(coords)
    edges = set()
    edge_points = []

    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        # Semiperimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        # Here's the radius filter.
        # print circum_r
        if circum_r < 1 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points


def set_coordinate_asap(coords_xml, order, x, y):
    coord_xml = ET.SubElement(coords_xml, 'Coordinate')
    coord_xml.set('Order', str(order))
    coord_xml.set('X', str(x))
    coord_xml.set('Y', str(y))


def create_asap_xml_from_coords(coords):
    root = ET.Element('ASAP_Annotations')
    annot_xml = ET.SubElement(root, 'Annotations')
    for j, coord_set in enumerate(coords):
        annot = ET.SubElement(annot_xml, 'Annotation')
        annot.set('Name', 'Annotation {}'.format(j))
        annot.set('Type', 'Polygon')
        annot.set('PartOfGroup', 'Region')
        annot.set('Color', '#F4FA58')
        coords_xml = ET.SubElement(annot, 'Coordinates')
        for i, point in enumerate(coord_set):
            set_coordinate_asap(coords_xml, i, point[1], point[0])
    groups_xml = ET.SubElement(root, 'AnnotationGroups')
    group_xml = ET.SubElement(groups_xml, 'Group')
    group_xml.set('Name', 'Region')
    group_xml.set('PartOfGroup', 'None')
    group_xml.set('Color', '#00ff00')
    ET.SubElement(group_xml, 'Attributes')
    return ET.ElementTree(root)


def calc_ratio(patch):
    ratio_patch = patch.copy()
    ratio_patch[ratio_patch > 1] = 1
    counts = np.unique(ratio_patch, return_counts=True)
    return (100 / counts[1][0]) * counts[1][1]


def concave_hull(
        input_file,
        output_dir,
        input_level=6,
        downsample=64,
        output_level=0,
        level_offset=0,
        alpha=0.07,
        min_size=1.5,
        bulk_class=1  # the tumour value in the seg mask
):
    wsi = open_multiresolutionimage_image(input_file)
    dimensions = wsi.getDimensions()
    res_zero = wsi.getSpacing()[0]

    # Read the mask at a reasonable level for fast processing.
    level = input_level
    wsi_width = int(dimensions[0] // downsample)
    wsi_height = int(dimensions[1] // downsample)
    spacing = res_zero * downsample

    # Ratio decides whether the approach for biopsies or resections is used.
    # A smaller kernel and min_size is used for biopsies.
    wsi_patch = wsi.getUCharPatch(
        startX=0, startY=0, width=wsi_width, height=wsi_height, level=level
    ).squeeze()
    ratio = calc_ratio(wsi_patch)
    wsi_patch = np.where(wsi_patch == bulk_class, wsi_patch, 0 * wsi_patch)
    min_size_px = mm2_to_px(1.0, spacing)
    kernel_diameter = dist_to_px(500, spacing)

    # applied fix for final leaderboard
    wsi_patch_indexes = skimage.morphology.remove_small_objects(
        ((wsi_patch == bulk_class)), min_size=mm2_to_px(0.005, spacing),
        connectivity=2
    )
    wsi_patch[wsi_patch_indexes == False] = 0
    kernel_diameter = dist_to_px(1000, spacing)
    min_size_px = mm2_to_px(min_size, spacing)

    print('spacing', spacing)
    print(f'min size in pixels {min_size_px}')
    print('ratio is:', ratio)
    print('wsi_dim', (wsi_width, wsi_height))
    print('kernel radius in pixels', kernel_diameter)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_diameter, kernel_diameter)
    )
    closing = cv2.morphologyEx(wsi_patch, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    wsi_patch = opening

    wsi_patch_indexes = skimage.morphology.remove_small_objects(
        ((wsi_patch == bulk_class)), min_size=min_size_px, connectivity=2)
    wsi_patch[wsi_patch_indexes == False] = 0

    points = np.argwhere(wsi_patch == bulk_class)
    if len(points) == 0:
        print(f'no hull found in {input_file} with indexes {bulk_class}')
        return
    concave_hull, edge_points = alpha_shape(points, alpha=alpha)

    if isinstance(
            concave_hull, shapely.geometry.polygon.Polygon) \
            or \
            isinstance(concave_hull, shapely.geometry.GeometryCollection):
        polygons = [concave_hull]
    else:
        polygons = list(concave_hull)

    # write polygons to annotations and add buffer
    buffersize = dist_to_px(250, spacing)
    coordinates = []
    for polygon in polygons:
        if polygon.area < min_size_px:
            continue
        polygon = polygon.buffer(buffersize)

        coordinates.append(
            [[x[0] * 2 ** (input_level + level_offset - output_level),
              x[1] * 2 ** (input_level + level_offset - output_level)]
             for x in polygon.boundary.coords[:-1]]
        )
    asap_annot = create_asap_xml_from_coords(coordinates)

    output_filename = os.path.basename(input_file)[:-4]
    asap_annot.write(os.path.join(output_dir, output_filename + ".xml"))
