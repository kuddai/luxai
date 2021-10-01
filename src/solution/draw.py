import _socket
import json

from solution.constants import UNIT_SPATIAL_DATA_RADIUS


class RewindClient():
    RED = 0xff0000
    GREEN = 0x00ff00
    BLUE = 0x0000ff
    DARK_RED = 0x770000
    DARK_GREEN = 0x007700
    DARK_BLUE = 0x000077
    TRANSPARENT = 0x7f000000
    INVISIBLE = 0x01000000

    def __init__(self, host=None, port=None):
        self._socket = _socket.socket()
        self._socket.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, True)
        if host is None:
            host = "0.0.0.0"
            port = 9111
        self._socket.connect((host, port))

    @staticmethod
    def _to_geojson(points):
        flat = []
        for p in points:
            flat.append(p[0])
            flat.append(p[1])
        return flat

    def _send(self, obj):
        if self._socket:
            self._socket.sendall(json.dumps(obj).encode('utf-8'))

    def line(self, x1, y1, x2, y2, color):
        self._send({
            'type': 'polyline',
            'points': [x1, y1, x2, y2],
            'color': color
        })

    def polyline(self, points, color):
        self._send({
            'type': 'polyline',
            'points': RewindClient._to_geojson(points),
            'color': color
        })

    def circle(self, x, y, radius, color, fill=False):
        self._send({
            'type': 'circle',
            'p': [x, y],
            'r': radius,
            'color': color,
            'fill': fill
        })

    def rectangle(self, x1, y1, x2, y2, color, fill=False):
        self._send({
            'type': 'rectangle',
            'tl': [x1, y1],
            'br': [x2, y2],
            'color': color,
            'fill': fill
        })

    def triangle(self, p1, p2, p3, color, fill=False):
        self._send({
            'type': 'triangle',
            'points': RewindClient._to_geojson([p1, p2, p3]),
            'color': color,
            'fill': fill
        })

    def circle_popup(self, x, y, radius, message):
        self._send({
            'type': 'popup',
            'p': [x, y],
            'r': radius,
            'text': message
        })

    def rect_popup(self, tl, br, message):
        self._send({
            'type': 'popup',
            'tl': RewindClient._to_geojson([tl]),
            'br': RewindClient._to_geojson([br]),
            'text': message
        })

    def message(self, msg):
        self._send({
            'type': 'message',
            'message': msg
        })

    def set_options(self, layer=None, permanent=None):
        data = {'type': 'options'}
        if layer is not None:
            data['layer'] = layer
        if permanent is not None:
            data['permanent'] = permanent
        self._send(data)

    def end_frame(self):
        self._send({'type': 'end'})


rewind_client = RewindClient()

def draw_unit_map(unit_map):
    colors = {
        0: 0x7f00cc00,
        1: 0x7f0000cc,
        2: 0x7fcc00cc,
        3: 0x7fcc0000,
        4: 0x7f00cccc,
        5: 0x7f333333,
    }
    for y in range(unit_map.shape[0]):
        for x in range(unit_map.shape[1]):
            for layer in range(unit_map.shape[2]):
                rewind_client.set_options(layer=layer + 1)
                rewind_client.rectangle(x * 30, y * 30, (x + 1) * 30, (y + 1) * 30, 0x333333)
                if unit_map[y][x][layer] and layer != 5:
                    rewind_client.rectangle(x * 30, y * 30, (x + 1) * 30, (y + 1) * 30, colors[layer], True)
                if unit_map[y][x][layer] == 0.0 and layer == 5:
                    rewind_client.rectangle(x * 30, y * 30, (x + 1) * 30, (y + 1) * 30, colors[layer], True)

    p = UNIT_SPATIAL_DATA_RADIUS + 0.5
    rewind_client.circle(p * 30, p * 30, UNIT_SPATIAL_DATA_RADIUS * 0.8, 0x333333, True)
