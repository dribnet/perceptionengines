import sys
import math
from PIL import Image, ImageDraw
import numpy as np
import csv

# canonical interpolation function, like https://p5js.org/reference/#/p5/map
def map_number(n, start1, stop1, start2, stop2):
  return ((n-start1)/(stop1-start1))*(stop2-start2)+start2;

# input: array of real vectors, length 8, each component normalized 0-1
def render(a, size):
    # split input array into header and rest
    header_length = 2
    head = a[:header_length]
    rest = a[header_length:]

    # determine background color from header
    R = int(map_number(head[0][0], 0, 1, 0, 255))
    G = int(map_number(head[0][1], 0, 1, 0, 255))
    B = int(map_number(head[0][2], 0, 1, 0, 255))

    # create the image and drawing context
    im = Image.new('RGB', (size, size), (R, G, B))
    draw = ImageDraw.Draw(im, 'RGB')

    # now draw lines
    min_width = 0.004 * size
    max_width = 0.04 * size
    for e in rest:
        w2 = int(map_number(e[4], 0, 1, min_width, max_width))
        # line width
        w = 2 * w2 + 2
        # line position
        x1 = map_number(e[0], 0, 1, w2, size-w2)
        y1 = map_number(e[1], 0, 1, w2, size-w2)
        x2 = map_number(e[2], 0, 1, w2, size-w2)
        y2 = map_number(e[3], 0, 1, w2, size-w2)

        # determine foreground color from header
        R = int(map_number(e[4], 0, 1, 0, 255))
        G = int(map_number(e[5], 0, 1, 0, 255))
        B = int(map_number(e[6], 0, 1, 0, 255))

        # draw line with round line caps (circles at the end)
        draw.line((x1, y1, x2, y2), fill=(R, G, B), width=w)
        draw.ellipse((x1-w2, y1-w2, x1+w2, y1+w2), fill=(R, G, B))
        draw.ellipse((x2-w2, y2-w2, x2+w2, y2+w2), fill=(R, G, B))

    return im
