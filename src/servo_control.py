# src/servo_control.py
"""
Sends pan angle commands (0-180) to the ESP8266 over serial.

Opening a serial port toggles DTR, which physically resets most ESP8266
boards (that's how Arduino IDE flashes them without a reset button). This
means every fresh connection needs real settle time before the board is
ready to receive commands - writing too early causes low-level Windows
write errors on CP210x/CH340 chips. This version waits long enough, flushes
stale buffers, and retries reconnects with backoff instead of giving up
after one failed attempt.
"""
from __future__ import annotations
import time
from typing import Optional

try:
    import serial
except ImportError:
    serial = None

BOOT_SETTLE_S = 4.0        # time to wait after opening the port for the board to finish rebooting
RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY_S = 1.5


class ServoController:
    def __init__(self, port: Optional[str] = None, baud: int = 9600):
        self.port = port
        self.baud = baud
        self.conn = None
        self.last_angle = 90
        self._connect()

    def _connect(self) -> bool:
        if not self.port or serial is None:
            print("[servo] no port given or pyserial missing -> PRINT-ONLY mode")
            return False

        for attempt in range(1, RECONNECT_ATTEMPTS + 1):
            try:
                conn = serial.Serial()
                conn.port = self.port
                conn.baudrate = self.baud
                conn.timeout = 1
                conn.xonxoff = False
                conn.rtscts = False
                conn.dsrdtr = False
                conn.open()

                print(f"[servo] opened {self.port} (attempt {attempt}), "
                      f"waiting {BOOT_SETTLE_S}s for board reboot...")
                time.sleep(BOOT_SETTLE_S)

                # clear out any boot-time garbage bytes sitting in the buffers
                conn.reset_input_buffer()
                conn.reset_output_buffer()

                self.conn = conn
                print(f"[servo] connected on {self.port}")
                return True

            except Exception as e:
                print(f"[servo] connect attempt {attempt} failed: {e}")
                if attempt < RECONNECT_ATTEMPTS:
                    time.sleep(RECONNECT_DELAY_S)

        print("[servo] all connection attempts failed -> PRINT-ONLY mode")
        self.conn = None
        return False

    def set_angle(self, angle: int):
        angle = max(0, min(180, int(angle)))
        if angle == self.last_angle:
            return
        self.last_angle = angle

        if self.conn is None:
            print(f"[servo] would send angle={angle}")
            return

        try:
            self.conn.write(f"{angle}\n".encode())
            print(f"[servo] SENT angle={angle}")
        except Exception as e:
            print(f"[servo] write failed ({e}); will attempt reconnect")
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
            self._connect()  # full retry-with-backoff sequence, not a single instant attempt

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass