import csv
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import pyqtgraph as pg
import serial
import serial.tools.list_ports
from PyQt6.QtCore import QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                             QHBoxLayout, QPushButton, QWidget, QLabel,
                             QComboBox, QSpinBox, QCheckBox, QGroupBox,
                             QGridLayout, QFileDialog)

BAUD_RATE = 115200

# Kart tarafındaki değerlerle aynı olmalı (main.c)
TIM_TICK_HZ = 250000
SLOTS = 3

MAX_SECONDS = 60
MAX_FPS = 200

# Nabız / SpO2 hesabı
VITALS_WINDOW_S = 8
PULSE_MIN_HZ = 0.7      # 42 atım/dk
PULSE_MAX_HZ = 3.5      # 210 atım/dk
MIN_SIGNAL = 200        # bunun altındaki sinyalde nabız dalgası gürültüde kaybolur
MIN_PEAK_RATIO = 4.0    # nabız tepesi / bant ortalaması


def actual_fps(fps):
    """Timer tam bölünmediği için karttaki gerçek hız istenenden biraz farklı."""
    arr = TIM_TICK_HZ // (SLOTS * fps)
    return TIM_TICK_HZ / (SLOTS * arr)


def pulse_band(x, fs):
    """Eğimi alır, sadece nabız bandını (0.7-3.5 Hz) bırakır."""
    n = len(x)
    idx = np.arange(n)
    x = x - np.polyval(np.polyfit(idx, x, 1), idx)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    spec[(freqs < PULSE_MIN_HZ) | (freqs > PULSE_MAX_HZ)] = 0
    return np.fft.irfft(spec, n)


def pulse_peak(x, fs):
    """Nabız bandındaki en güçlü frekansı ve bant ortalamasına oranını döndürür."""
    n = len(x)
    nfft = 8 * n
    amp = np.abs(np.fft.rfft(x * np.hanning(n), nfft))
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    band = (freqs >= PULSE_MIN_HZ) & (freqs <= PULSE_MAX_HZ)
    amp_band = amp[band]
    k = int(np.argmax(amp_band))
    mean = amp_band.mean()
    ratio = amp_band[k] / mean if mean > 0 else 0.0
    return freqs[band][k], ratio


def estimate_vitals(red, ir, fs):
    """
    red, ir: karanlığı çıkarılmış sinyaller (LED1 = kırmızı, LED2 = kızılötesi).
    (nabız, spo2, mesaj, güvenilir) döndürür.

    Sinyal zayıf olsa da hesap yapılır; o durumda güvenilir False gelir ve
    sayılar gürültüden çıkmış olabilir. Hiç hesaplanamayan değer None olur.
    """
    need = int(VITALS_WINDOW_S * fs)
    if len(red) < need:
        return None, None, "Veri toplanıyor...", False

    red = np.asarray(red[-need:], dtype=float)
    ir = np.asarray(ir[-need:], dtype=float)

    dc_red = red.mean()
    dc_ir = ir.mean()

    ac_red = pulse_band(red, fs)
    ac_ir = pulse_band(ir, fs)

    f_ir, ratio_ir = pulse_peak(ac_ir, fs)
    f_red, ratio_red = pulse_peak(ac_red, fs)
    bpm = f_ir * 60.0

    rms_red = ac_red.std()
    rms_ir = ac_ir.std()

    # Oran için iki sinyalin de pozitif olması gerek
    spo2 = None
    r = None
    if dc_red > 0 and dc_ir > 0 and rms_ir > 0:
        r = (rms_red / dc_red) / (rms_ir / dc_ir)
        # Yaygın kullanılan doğrusal yaklaşım; bu LED'ler için kalibre edilmedi
        spo2 = min(100.0, max(0.0, 110.0 - 25.0 * r))

    if dc_red < MIN_SIGNAL or dc_ir < MIN_SIGNAL:
        msg = f"Sinyal zayıf (LED1: {dc_red:.0f}, LED2: {dc_ir:.0f}, en az {MIN_SIGNAL})"
        return bpm, spo2, msg, False

    # Gerçek nabızda iki kanal aynı frekansta tepe yapar
    if ratio_ir < MIN_PEAK_RATIO or ratio_red < MIN_PEAK_RATIO or abs(f_ir - f_red) > 0.15:
        return bpm, spo2, "Nabız net değil (parmağı sabit tut)", False

    return bpm, spo2, f"R = {r:.2f}", True


class SerialWorker(QThread):
    frames_received = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, port, fps, led_mask):
        super().__init__()
        self.port = port
        self.fps = fps
        self.led_mask = led_mask
        self.running = False
        self.ser = None

    def send(self, cmd):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write((cmd + "\n").encode())
            except serial.SerialException as e:
                self.error_occurred.emit(str(e))

    def run(self):
        self.running = True
        try:
            self.ser = serial.Serial(self.port, BAUD_RATE, timeout=0.05)
            QThread.msleep(200)

            self.send("STOP")
            QThread.msleep(50)
            self.ser.reset_input_buffer()

            self.send(f"RATE,{self.fps}")
            self.send(f"LED,{self.led_mask}")
            self.send("START")

            pending = b""
            last_data = time.time()

            while self.running:
                chunk = self.ser.read(256)
                if not chunk:
                    if time.time() - last_data > 2.0:
                        self.error_occurred.emit("Karttan veri gelmiyor")
                        last_data = time.time()
                    continue

                last_data = time.time()
                pending += chunk
                lines = pending.split(b"\n")
                pending = lines.pop()

                frames = []
                for line in lines:
                    parts = line.strip().split(b",")
                    if len(parts) != 3:
                        continue
                    try:
                        frames.append(tuple(int(p) for p in parts))
                    except ValueError:
                        continue

                if frames:
                    self.frames_received.emit(frames)

            self.send("STOP")
            self.ser.flush()

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("O2 Fotodiyot Monitörü")
        self.resize(1100, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Ayarlar Paneli
        settings_group = QGroupBox("Ayarlar")
        settings_layout = QGridLayout()

        settings_layout.addWidget(QLabel("Port:"), 0, 0)
        self.combo_port = QComboBox()
        settings_layout.addWidget(self.combo_port, 0, 1)
        self.btn_refresh = QPushButton("Yenile")
        self.btn_refresh.clicked.connect(self.refresh_ports)
        settings_layout.addWidget(self.btn_refresh, 0, 2)

        settings_layout.addWidget(QLabel("Hız (ölçüm/sn):"), 0, 3)
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(10, MAX_FPS)
        self.spin_fps.setValue(83)
        settings_layout.addWidget(self.spin_fps, 0, 4)

        settings_layout.addWidget(QLabel("Pencere (sn):"), 0, 5)
        self.spin_window = QSpinBox()
        self.spin_window.setRange(1, MAX_SECONDS)
        self.spin_window.setValue(10)
        settings_layout.addWidget(self.spin_window, 0, 6)

        # LED'ler çalışırken de açılıp kapatılabiliyor
        self.chk_led1 = QCheckBox("LED1 (PE2)")
        self.chk_led1.setChecked(True)
        self.chk_led1.toggled.connect(self.send_led_mask)
        self.chk_led2 = QCheckBox("LED2 (PE3)")
        self.chk_led2.setChecked(True)
        self.chk_led2.toggled.connect(self.send_led_mask)
        settings_layout.addWidget(self.chk_led1, 1, 0, 1, 2)
        settings_layout.addWidget(self.chk_led2, 1, 2, 1, 2)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Grafikte neyin çizileceği LED kutularından bağımsız
        plot_group = QGroupBox("Grafikte Göster")
        plot_layout = QHBoxLayout()
        self.chk_show_sig1 = QCheckBox("LED1 - karanlık")
        self.chk_show_sig2 = QCheckBox("LED2 - karanlık")
        self.chk_show_dark = QCheckBox("Karanlık")
        self.chk_show_raw1 = QCheckBox("LED1 ham")
        self.chk_show_raw2 = QCheckBox("LED2 ham")
        self.chk_show_sig1.setChecked(True)
        self.chk_show_sig2.setChecked(True)
        for chk in (self.chk_show_sig1, self.chk_show_sig2, self.chk_show_dark,
                    self.chk_show_raw1, self.chk_show_raw2):
            chk.toggled.connect(self.update_curve_visibility)
            plot_layout.addWidget(chk)
        plot_layout.addStretch()
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        # Kontrol Bölümü
        controls_layout = QHBoxLayout()

        self.btn_start = QPushButton("BAŞLAT")
        self.btn_start.clicked.connect(self.start_acquisition)
        self.btn_start.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white; padding: 10px;")

        self.btn_stop = QPushButton("DURDUR")
        self.btn_stop.clicked.connect(self.stop_acquisition)
        self.btn_stop.setStyleSheet("font-weight: bold; background-color: #F44336; color: white; padding: 10px;")
        self.btn_stop.setEnabled(False)

        self.btn_freeze = QPushButton("SABİTLE")
        self.btn_freeze.setCheckable(True)
        self.btn_freeze.setStyleSheet("font-weight: bold; background-color: #2196F3; color: white; padding: 10px;")

        self.btn_save = QPushButton("KAYDET (CSV)")
        self.btn_save.clicked.connect(self.save_csv)
        self.btn_save.setStyleSheet("font-weight: bold; background-color: #607D8B; color: white; padding: 10px;")

        self.btn_vitals = QPushButton("NABIZ / SpO2")
        self.btn_vitals.setCheckable(True)
        self.btn_vitals.toggled.connect(self.toggle_vitals)
        self.btn_vitals.setStyleSheet("font-weight: bold; background-color: #9C27B0; color: white; padding: 10px;")

        self.status_label = QLabel("Durum: Hazır")
        self.status_label.setStyleSheet("font-weight: bold;")

        self.value_label = QLabel("")
        self.value_label.setStyleSheet("font-weight: bold; color: blue;")

        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_stop)
        controls_layout.addWidget(self.btn_freeze)
        controls_layout.addWidget(self.btn_save)
        controls_layout.addWidget(self.btn_vitals)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch()
        controls_layout.addWidget(self.value_label)

        layout.addLayout(controls_layout)

        # Nabız / SpO2 göstergesi (butonla açılır)
        self.vitals_group = QGroupBox("Nabız / SpO2  (LED1 = kırmızı, LED2 = kızılötesi)")
        vitals_layout = QHBoxLayout()
        self.bpm_label = QLabel("Nabız: -- atım/dk")
        self.bpm_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #D32F2F;")
        self.spo2_label = QLabel("SpO2: -- %")
        self.spo2_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #1976D2;")
        self.vitals_status = QLabel("")
        vitals_layout.addWidget(self.bpm_label)
        vitals_layout.addSpacing(40)
        vitals_layout.addWidget(self.spo2_label)
        vitals_layout.addSpacing(40)
        vitals_layout.addWidget(self.vitals_status)
        vitals_layout.addStretch()
        self.vitals_group.setLayout(vitals_layout)
        self.vitals_group.hide()
        layout.addWidget(self.vitals_group)

        self.vitals_timer = QTimer()
        self.vitals_timer.timeout.connect(self.update_vitals)

        # Grafik
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()
        self.plot_widget.setLabel('left', 'ADC Değeri')
        self.plot_widget.setLabel('bottom', 'Zaman (sn)')

        self.curve_sig1 = self.plot_widget.plot(pen=pg.mkPen((255, 0, 0), width=1.5), name="LED1 - karanlık")
        self.curve_sig2 = self.plot_widget.plot(pen=pg.mkPen((0, 0, 255), width=1.5), name="LED2 - karanlık")
        self.curve_dark = self.plot_widget.plot(pen=pg.mkPen((120, 120, 120), width=1), name="Karanlık")
        self.curve_raw1 = self.plot_widget.plot(pen=pg.mkPen((255, 165, 0), width=1), name="LED1 ham")
        self.curve_raw2 = self.plot_widget.plot(pen=pg.mkPen((0, 128, 0), width=1), name="LED2 ham")

        layout.addWidget(self.plot_widget)

        buf_len = MAX_SECONDS * MAX_FPS
        self.buf_dark = deque(maxlen=buf_len)
        self.buf_led1 = deque(maxlen=buf_len)
        self.buf_led2 = deque(maxlen=buf_len)

        # BAŞLAT'tan beri gelen her şey; grafik penceresinden bağımsız
        self.record = []

        self.worker = None
        self.fps = actual_fps(self.spin_fps.value())
        self.frame_count = 0
        self.start_time = 0

        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plot)

        self.refresh_ports()
        self.update_curve_visibility()

    @pyqtSlot()
    def refresh_ports(self):
        self.combo_port.clear()
        stlink_index = -1
        for i, p in enumerate(serial.tools.list_ports.comports()):
            self.combo_port.addItem(f"{p.device} - {p.description}", p.device)
            if "STLink" in p.description or "STMicroelectronics" in p.description:
                stlink_index = i
        if stlink_index >= 0:
            self.combo_port.setCurrentIndex(stlink_index)

    def led_mask(self):
        mask = 0
        if self.chk_led1.isChecked():
            mask |= 1
        if self.chk_led2.isChecked():
            mask |= 2
        return mask

    @pyqtSlot()
    def send_led_mask(self):
        if self.worker and self.worker.isRunning():
            self.worker.send(f"LED,{self.led_mask()}")

    @pyqtSlot()
    def update_curve_visibility(self):
        self.curve_sig1.setVisible(self.chk_show_sig1.isChecked())
        self.curve_sig2.setVisible(self.chk_show_sig2.isChecked())
        self.curve_dark.setVisible(self.chk_show_dark.isChecked())
        self.curve_raw1.setVisible(self.chk_show_raw1.isChecked())
        self.curve_raw2.setVisible(self.chk_show_raw2.isChecked())

    @pyqtSlot()
    def start_acquisition(self):
        port = self.combo_port.currentData()
        if port is None:
            self.show_error("Port seçilmedi!")
            return

        if self.worker is None or not self.worker.isRunning():
            self.buf_dark.clear()
            self.buf_led1.clear()
            self.buf_led2.clear()
            self.record = []

            fps = self.spin_fps.value()
            self.fps = actual_fps(fps)

            self.worker = SerialWorker(port, fps, self.led_mask())
            self.worker.frames_received.connect(self.on_frames)
            self.worker.error_occurred.connect(self.show_error)
            self.worker.finished.connect(self.on_worker_finished)
            self.worker.start()

            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.settings_enabled(False)
            self.status_label.setText("Durum: Okunuyor...")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")

            self.start_time = time.time()
            self.frame_count = 0
            self.plot_timer.start(33)

    @pyqtSlot()
    def stop_acquisition(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()

        self.plot_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.settings_enabled(True)
        self.status_label.setText("Durum: Durduruldu")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")

    @pyqtSlot()
    def on_worker_finished(self):
        # Port açılamazsa ya da kablo çıkarsa thread kendiliğinden biter
        if self.btn_stop.isEnabled():
            self.plot_timer.stop()
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.settings_enabled(True)

    @pyqtSlot(bool)
    def toggle_vitals(self, checked):
        self.vitals_group.setVisible(checked)
        if checked:
            self.update_vitals()
            self.vitals_timer.start(500)
        else:
            self.vitals_timer.stop()

    @pyqtSlot()
    def update_vitals(self):
        bpm, spo2, msg, reliable = None, None, "Ölçüm çalışmıyor", False

        if self.worker and self.worker.isRunning():
            if self.led_mask() != 3:
                msg = "İki LED de açık olmalı"
            else:
                need = int(VITALS_WINDOW_S * self.fps)
                dark = np.array(self.buf_dark, dtype=float)[-need:]
                led1 = np.array(self.buf_led1, dtype=float)[-need:]
                led2 = np.array(self.buf_led2, dtype=float)[-need:]
                if len(led1) and max(led1.max(), led2.max()) >= 4090:
                    msg = "Sinyal doymuş (4095), R3'ü küçült"
                else:
                    bpm, spo2, msg, reliable = estimate_vitals(led1 - dark, led2 - dark, self.fps)

        # Güvenilmez değerler de yazılır, ama gri ve "~" ile
        mark = "" if reliable else "~"
        self.bpm_label.setText(f"Nabız: {mark}{bpm:.0f} atım/dk" if bpm is not None else "Nabız: -- atım/dk")
        self.spo2_label.setText(f"SpO2: {mark}{spo2:.0f} %" if spo2 is not None else "SpO2: -- %")
        self.bpm_label.setStyleSheet(
            f"font-size: 22px; font-weight: bold; color: {'#D32F2F' if reliable else '#9E9E9E'};")
        self.spo2_label.setStyleSheet(
            f"font-size: 22px; font-weight: bold; color: {'#1976D2' if reliable else '#9E9E9E'};")
        if bpm is not None and not reliable:
            msg += " - değerler güvenilir değil"
        self.vitals_status.setText(msg)

    @pyqtSlot()
    def save_csv(self):
        if not self.record:
            self.show_error("Kaydedilecek veri yok")
            return

        # Ölçüm sürerken basılırsa o ana kadarki kısmı al
        rows = list(self.record)
        fps = self.fps

        default_name = datetime.now().strftime("o2_%Y%m%d_%H%M%S.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Veriyi Kaydet", default_name, "CSV (*.csv)")
        if not path:
            return

        # Örnekler eşit aralıklı; zaman sütunu örnek numarası / hız
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["t_s", "dark", "led1", "led2", "sig1", "sig2", "led1_on", "led2_on"])
                for i, (dark, led1, led2, mask) in enumerate(rows):
                    writer.writerow([f"{i / fps:.6f}", dark, led1, led2,
                                     led1 - dark, led2 - dark, mask & 1, (mask >> 1) & 1])
        except OSError as e:
            self.show_error(str(e))
            return

        self.status_label.setText(f"Kaydedildi: {len(rows)} ölçüm, {len(rows) / fps:.1f} sn")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")

    def settings_enabled(self, state):
        self.combo_port.setEnabled(state)
        self.btn_refresh.setEnabled(state)
        self.spin_fps.setEnabled(state)

    @pyqtSlot(list)
    def on_frames(self, frames):
        mask = self.led_mask()
        for dark, led1, led2 in frames:
            self.buf_dark.append(dark)
            self.buf_led1.append(led1)
            self.buf_led2.append(led2)
            self.record.append((dark, led1, led2, mask))
        self.frame_count += len(frames)

    @pyqtSlot()
    def update_plot(self):
        if self.btn_freeze.isChecked() or len(self.buf_dark) == 0:
            return

        n = min(len(self.buf_dark), int(self.spin_window.value() * self.fps))
        dark = np.array(self.buf_dark, dtype=np.int32)[-n:]
        led1 = np.array(self.buf_led1, dtype=np.int32)[-n:]
        led2 = np.array(self.buf_led2, dtype=np.int32)[-n:]
        t = np.arange(n) / self.fps

        sig1 = led1 - dark
        sig2 = led2 - dark

        self.curve_sig1.setData(t, sig1)
        self.curve_sig2.setData(t, sig2)
        self.curve_dark.setData(t, dark)
        self.curve_raw1.setData(t, led1)
        self.curve_raw2.setData(t, led2)

        elapsed = time.time() - self.start_time
        rate = self.frame_count / elapsed if elapsed > 0 else 0
        self.value_label.setText(
            f"Karanlık: {dark[-1]} | LED1: {sig1[-1]} | LED2: {sig2[-1]} | {rate:.0f} ölçüm/sn")

    @pyqtSlot(str)
    def show_error(self, err_msg):
        self.status_label.setText(f"Hata: {err_msg}")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
