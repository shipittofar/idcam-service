#!/usr/bin/env python3
"""Stand-alone WS2812B strip check for Raspberry Pi 5 (no idCam config needed).

Cycles through the colours the idCam status indicator uses. Stop the idcam
service first so both do not drive the strip at once:

    sudo systemctl stop idcam
    /opt/idcam/venv/bin/python scripts/ws2812_test.py
"""

import sys
import time

import adafruit_pixelbuf
import board
from adafruit_raspberry_pi5_neopixel_write import neopixel_write

# ===== Configuration =====
LED_COUNT = 30
LED_PIN = board.D18
BRIGHTNESS = 0.2
DELAY = 2


class Pi5Pixelbuf(adafruit_pixelbuf.PixelBuf):
    def __init__(self, pin, size, **kwargs):
        self._pin = pin
        super().__init__(size=size, **kwargs)

    def _transmit(self, buf):
        neopixel_write(self._pin, buf)


def main():
    print("================================")
    print(" Raspberry Pi 5 - WS2812B Test")
    print("================================")
    print(f"LED Count: {LED_COUNT}")
    print("Data Pin: GPIO18 (Physical Pin 12)")
    print(f"Brightness: {int(BRIGHTNESS * 100)}%")
    print("Initializing LED strip...", flush=True)

    pixels = Pi5Pixelbuf(
        LED_PIN,
        LED_COUNT,
        auto_write=False,
        byteorder="GRB",
        brightness=BRIGHTNESS,
    )

    colors = [
        ("Yellow", (255, 255, 0)),
        ("Blue", (0, 0, 255)),
        ("White", (255, 255, 255)),
        ("Red", (255, 0, 0)),
        ("Green", (0, 255, 0)),
    ]

    try:
        print("Initialization complete.")
        print("Press Ctrl+C to stop.\n", flush=True)

        while True:
            for name, color in colors:
                print(f"Status: Displaying {name} | RGB={color}", flush=True)
                pixels.fill(color)
                pixels.show()
                time.sleep(DELAY)

    except KeyboardInterrupt:
        print("\nStatus: Stop signal received.", flush=True)

    finally:
        print("Status: Turning off all LEDs...", flush=True)
        try:
            pixels.fill((0, 0, 0))
            pixels.show()
            time.sleep(0.1)
            print("Status: LED strip turned off.", flush=True)
        except Exception as e:
            print(f"Error while turning off LEDs: {e}", flush=True)

        print("Status: Program terminated.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
