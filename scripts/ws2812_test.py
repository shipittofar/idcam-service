#!/usr/bin/env python3
"""Stand-alone WS2812B strip check for Raspberry Pi 5 (no idCam config needed).

Cycles through the colours the idCam status indicator uses, naming what each means. Stop the
idcam service first so both do not drive the strip at once:

    sudo systemctl stop idcam
    /opt/idcam/venv/bin/python scripts/ws2812_test.py                 # GPIO18, 30 LEDs
    /opt/idcam/venv/bin/python scripts/ws2812_test.py --pin 12 --count 60
    /opt/idcam/venv/bin/python scripts/ws2812_test.py --config config.yaml   # use its led: block
"""

import argparse
import sys
import time

import adafruit_pixelbuf
import board
from adafruit_raspberry_pi5_neopixel_write import neopixel_write

# ===== Defaults (override with the options above) =====
LED_COUNT = 30
LED_PIN_NUMBER = 18
BRIGHTNESS = 0.2
DELAY = 2


def parse_args():
    parser = argparse.ArgumentParser(description="WS2812B strip test")
    parser.add_argument("--pin", type=int, help="BCM data pin (default 18 = physical pin 12)")
    parser.add_argument("--count", type=int, help="number of LEDs")
    parser.add_argument("--brightness", type=float, help="0.0 - 1.0")
    parser.add_argument("--config", help="read pin/count/brightness/byteorder from the led: block of this YAML")
    return parser.parse_args()


def load_settings(args):
    settings = {"pin": LED_PIN_NUMBER, "count": LED_COUNT, "brightness": BRIGHTNESS, "byteorder": "GRB"}
    if args.config:
        import yaml

        with open(args.config, encoding="utf-8") as fh:
            led = (yaml.safe_load(fh) or {}).get("led") or {}
        settings["pin"] = int(led.get("gpio_pin", settings["pin"]))
        settings["count"] = int(led.get("count", settings["count"]))
        settings["brightness"] = float(led.get("brightness", settings["brightness"]))
        settings["byteorder"] = str(led.get("byteorder", settings["byteorder"]))
    if args.pin is not None:
        settings["pin"] = args.pin
    if args.count is not None:
        settings["count"] = args.count
    if args.brightness is not None:
        settings["brightness"] = args.brightness
    return settings


class Pi5Pixelbuf(adafruit_pixelbuf.PixelBuf):
    def __init__(self, pin, size, **kwargs):
        self._pin = pin
        super().__init__(size=size, **kwargs)

    def _transmit(self, buf):
        neopixel_write(self._pin, buf)


def main():
    cfg = load_settings(parse_args())
    pin = getattr(board, f"D{cfg['pin']}")
    print("================================")
    print(" Raspberry Pi 5 - WS2812B Test")
    print("================================")
    print(f"LED Count: {cfg['count']}")
    print(f"Data Pin: GPIO{cfg['pin']}")
    print(f"Brightness: {int(cfg['brightness'] * 100)}%")
    print("Initializing LED strip...", flush=True)

    pixels = Pi5Pixelbuf(
        pin,
        cfg["count"],
        auto_write=False,
        byteorder=cfg["byteorder"],
        brightness=cfg["brightness"],
    )

    # (meaning, colour, blinking)
    colors = [
        ("White          = ready: place the document (also lights it while it is captured)", (255, 255, 255), False),
        ("Blue blinking  = captured: checking and uploading, wait", (0, 0, 255), True),
        ("Green          = uploaded (for a moment)", (0, 255, 0), False),
        ("Green blinking = done: remove the document", (0, 255, 0), True),
        ("White blinking = failed again: remove the document, then place it again", (255, 255, 255), True),
        ("Red            = capture or upload failed (short flash) / error (blinking)", (255, 0, 0), False),
        ("Yellow blinking = recovering", (255, 255, 0), True),
    ]

    try:
        print("Initialization complete.")
        print("Press Ctrl+C to stop.\n", flush=True)

        while True:
            for name, color, blinking in colors:
                print(f"Status: Displaying {name} | RGB={color}", flush=True)
                if not blinking:
                    pixels.fill(color)
                    pixels.show()
                    time.sleep(DELAY)
                    continue
                end = time.monotonic() + DELAY
                while time.monotonic() < end:
                    for level in (color, (0, 0, 0)):
                        pixels.fill(level)
                        pixels.show()
                        time.sleep(0.4)

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
