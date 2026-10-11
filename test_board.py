"""
Kart testi. Kullanım:  python test_board.py [PORT]

Sırayla şunlara bakar:
  1. START sonrası veri geliyor mu, satır biçimi doğru mu
  2. RATE komutu ölçüm hızını değiştiriyor mu
  3. LED komutu LED'leri tek tek açıp kapatıyor mu
  4. STOP sonrası veri kesiliyor mu
"""
import sys
import time

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM6"
BAUD_RATE = 115200

TIM_TICK_HZ = 250000
SLOTS = 3

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'GECTI' if ok else 'KALDI'}] {name}" + (f"  ({detail})" if detail else ""))


def send(ser, cmd):
    ser.write((cmd + "\n").encode())


def read_frames(ser, seconds):
    """Verilen süre boyunca okur, (geçerli satırlar, bozuk satır sayısı) döndürür."""
    frames = []
    bad = 0
    pending = b""
    end = time.time() + seconds
    while time.time() < end:
        pending += ser.read(256)
        lines = pending.split(b"\n")
        pending = lines.pop()
        for line in lines:
            parts = line.strip().split(b",")
            try:
                if len(parts) != 3:
                    raise ValueError
                frames.append(tuple(int(p) for p in parts))
            except ValueError:
                bad += 1
    return frames, bad


def settle(ser):
    # komuttan sonra yarım kalan satırları at
    time.sleep(0.15)
    ser.reset_input_buffer()
    ser.readline()


def mean_signals(frames):
    n = len(frames)
    dark = sum(f[0] for f in frames) / n
    sig1 = sum(f[1] - f[0] for f in frames) / n
    sig2 = sum(f[2] - f[0] for f in frames) / n
    return dark, sig1, sig2


def main():
    print(f"Port: {PORT}")
    ser = serial.Serial(PORT, BAUD_RATE, timeout=0.05)
    time.sleep(0.3)

    send(ser, "STOP")
    send(ser, "LED,3")
    send(ser, "RATE,83")
    time.sleep(0.2)
    ser.reset_input_buffer()

    # 1. Veri akışı
    print("\n-- Veri akışı --")
    send(ser, "START")
    settle(ser)
    frames, bad = read_frames(ser, 1.0)
    check("START sonrası veri geliyor", len(frames) > 0, f"{len(frames)} satır")
    if not frames:
        print("\nKarttan veri gelmedi. Kontrol et:")
        print("  - Program çalışıyor mu (debug'da duraklatılmış olmasın)")
        print("  - Adaptör RX -> PA9, adaptör TX -> PA10, GND ortak")
        ser.close()
        return 1
    check("Satır biçimi doğru", bad <= 1, f"{bad} bozuk satır")
    in_range = all(0 <= v <= 4095 for f in frames for v in f)
    check("Değerler 0-4095 arasında", in_range)

    # 2. Hız
    print("\n-- Ölçüm hızı --")
    for fps in (20, 83, 200):
        send(ser, f"RATE,{fps}")
        settle(ser)
        frames, _ = read_frames(ser, 2.0)
        expected = TIM_TICK_HZ / (SLOTS * (TIM_TICK_HZ // (SLOTS * fps)))
        measured = len(frames) / 2.0
        ok = abs(measured - expected) <= max(2.0, expected * 0.05)
        check(f"RATE,{fps}", ok, f"beklenen {expected:.1f}, ölçülen {measured:.1f}")
    send(ser, "RATE,83")

    # 3. LED kontrolü
    print("\n-- LED kontrolü --")
    readings = {}
    for mask in (0, 1, 2, 3):
        send(ser, f"LED,{mask}")
        settle(ser)
        frames, _ = read_frames(ser, 1.0)
        readings[mask] = mean_signals(frames)
        dark, sig1, sig2 = readings[mask]
        print(f"   LED,{mask}: karanlık={dark:7.1f}  sinyal1={sig1:7.1f}  sinyal2={sig2:7.1f}")

    # kapalı LED'in sinyali sıfıra yakın olmalı
    tol = 5.0
    check("LED,0: iki sinyal de sıfır", abs(readings[0][1]) < tol and abs(readings[0][2]) < tol)
    check("LED,1: sinyal2 sıfır", abs(readings[1][2]) < tol)
    check("LED,2: sinyal1 sıfır", abs(readings[2][1]) < tol)

    # açık LED'in sinyali kapalı halinden yüksek olmalı (ışık sensöre ulaşıyorsa)
    gain1 = readings[1][1] - readings[0][1]
    gain2 = readings[2][2] - readings[0][2]
    check("LED1 ışığı sensöre ulaşıyor", gain1 > tol, f"fark {gain1:.1f}")
    check("LED2 ışığı sensöre ulaşıyor", gain2 > tol, f"fark {gain2:.1f}")

    # 4. Durdurma
    print("\n-- Durdurma --")
    send(ser, "STOP")
    settle(ser)
    frames, _ = read_frames(ser, 0.5)
    check("STOP sonrası veri kesiliyor", len(frames) == 0, f"{len(frames)} satır")

    send(ser, "LED,3")
    ser.close()

    passed = sum(results)
    print(f"\nSonuç: {passed}/{len(results)} test geçti")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
