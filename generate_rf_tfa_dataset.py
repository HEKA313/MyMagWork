"""
Этот файл предназначен для генерации синтетического датасета радиосигналов. Он умеет создавать:
— LPI-сигналы;
— расширенные радиолокационные классы;
— цифровые коммуникационные сигналы;
— аналоговые коммуникационные сигналы;
— шумовые и немодулированные сигналы.
На выходе файл может создавать изображения частотно-временных портретов.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.signal import hilbert, lfilter, resample, stft

# Базовый набор LPI-сигналов, используемый для основной задачи классификации
CORE_LPI_CLASSES: Tuple[str, ...] = (
	"Rect", "LFM", "Barker", "Costas", "Frank",
	"P1", "P2", "P3", "P4", "T1", "T2", "T3", "T4",
)

# Расширенный набор радиолокационных сигналов: базовые LPI + дополнительные модели
EXTENDED_RADAR_CLASSES: Tuple[str, ...] = CORE_LPI_CLASSES + (
	"NLFM", "MLS", "Legendre", "OFDM_Radar",
	"FMCW_Up", "FMCW_Down", "FMCW_Triangular",
	"FH_FSK", "SFCW", "NoiseOnly", "Unmodulated_CW",
)

# Набор цифровых коммуникационных модуляций для смешанного RF-датасета
COMM_DIGITAL_CLASSES: Tuple[str, ...] = (
	"BPSK_Random", "QPSK", "8PSK", "16QAM", "64QAM",
	"PAM4", "GFSK", "CPFSK",
)

# Набор аналоговых коммуникационных сигналов
COMM_ANALOG_CLASSES: Tuple[str, ...] = (
	"B_FM", "DSB_AM", "SSB_AM",
)

# Полный смешанный набор классов: радарные, цифровые и аналоговые сигналы
MIXED_RF_CLASSES: Tuple[str, ...] = EXTENDED_RADAR_CLASSES + COMM_DIGITAL_CLASSES + COMM_ANALOG_CLASSES

# Профили генерации: определяют набор классов для датасета
PROFILE_CLASSES: Dict[str, Tuple[str, ...]] = {
	"core_lpi": CORE_LPI_CLASSES,
	"extended_lpi": EXTENDED_RADAR_CLASSES,
	"mixed_rf": MIXED_RF_CLASSES,
}

# Сопоставление каждого класса с доменом и семейством для metadata.csv
CLASS_INFO: Dict[str, Dict[str, str]] = {
	"Rect": {"domain": "radar_lpi", "family": "simple_radar"},
	"LFM": {"domain": "radar_lpi", "family": "fm_radar"},
	"Barker": {"domain": "radar_lpi", "family": "binary_phase_code"},
	"Costas": {"domain": "radar_lpi", "family": "frequency_hopping"},
	"Frank": {"domain": "radar_lpi", "family": "polyphase"},
	"P1": {"domain": "radar_lpi", "family": "polyphase"},
	"P2": {"domain": "radar_lpi", "family": "polyphase"},
	"P3": {"domain": "radar_lpi", "family": "polyphase"},
	"P4": {"domain": "radar_lpi", "family": "polyphase"},
	"T1": {"domain": "radar_lpi", "family": "polytime"},
	"T2": {"domain": "radar_lpi", "family": "polytime"},
	"T3": {"domain": "radar_lpi", "family": "polytime"},
	"T4": {"domain": "radar_lpi", "family": "polytime"},
	"NLFM": {"domain": "radar_lpi", "family": "fm_radar"},
	"MLS": {"domain": "radar_lpi", "family": "binary_phase_code"},
	"Legendre": {"domain": "radar_lpi", "family": "binary_phase_code"},
	"OFDM_Radar": {"domain": "radar_lpi", "family": "multicarrier"},
	"FMCW_Up": {"domain": "radar_lpi", "family": "fm_radar"},
	"FMCW_Down": {"domain": "radar_lpi", "family": "fm_radar"},
	"FMCW_Triangular": {"domain": "radar_lpi", "family": "fm_radar"},
	"FH_FSK": {"domain": "radar_lpi", "family": "frequency_hopping"},
	"SFCW": {"domain": "radar_lpi", "family": "frequency_hopping"},
	"NoiseOnly": {"domain": "noise_or_unmodulated", "family": "noise_or_unmodulated"},
	"Unmodulated_CW": {"domain": "noise_or_unmodulated", "family": "noise_or_unmodulated"},
	"BPSK_Random": {"domain": "comm_digital", "family": "digital_psk"},
	"QPSK": {"domain": "comm_digital", "family": "digital_psk"},
	"8PSK": {"domain": "comm_digital", "family": "digital_psk"},
	"16QAM": {"domain": "comm_digital", "family": "digital_qam"},
	"64QAM": {"domain": "comm_digital", "family": "digital_qam"},
	"PAM4": {"domain": "comm_digital", "family": "digital_pam"},
	"GFSK": {"domain": "comm_digital", "family": "digital_fsk"},
	"CPFSK": {"domain": "comm_digital", "family": "digital_fsk"},
	"B_FM": {"domain": "comm_analog", "family": "analog_fm"},
	"DSB_AM": {"domain": "comm_analog", "family": "analog_am"},
	"SSB_AM": {"domain": "comm_analog", "family": "analog_am"},
}

# Таблица известных бинарных кодов Баркера
BARKER_CODES: Dict[int, Sequence[int]] = {
	2: [1, -1],
	3: [1, 1, -1],
	4: [1, 1, -1, 1],
	5: [1, 1, 1, -1, 1],
	7: [1, 1, 1, -1, -1, 1, -1],
	11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
	13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
}

# Отводы LFSR для генерации MLS-последовательностей разных порядков
MLS_TAPS: Dict[int, Tuple[int, ...]] = {
	5: (5, 2),
	6: (6, 1),
	7: (7, 1),
	8: (8, 6, 5, 4),
}


# Параметры построения STFT-спектрограммы
@dataclass(frozen=True)
class StftConfig:
	nperseg: int = 128  # длина окна STFT
	noverlap: int = 112  # перекрытие соседних окон
	nfft: int = 256  # число точек БПФ
	dynamic_range_db: float = 55.0  # динамический диапазон изображения в дБ
	image_size: int = 224  # итоговый размер изображения


# Параметры практической реализации распределения Чои–Вильямса
@dataclass(frozen=True)
class CwdConfig:
	sigma: float = 1.0  # параметр сглаживающего ядра
	max_lag: int = 96  # максимальная задержка
	nfft: int = 256  # размер БПФ по оси
	max_samples: int = 1024  # максимальное число отсчётов
	dynamic_range_db: float = 55.0  # динамический диапазон изображения
	image_size: int = 224  # размер итоговой картинки


# Общие параметры генерации IQ-сигналов и аппаратно-канальных искажений
@dataclass(frozen=True)
class GeneratorConfig:
	fs: float = 100e6  # частота дискретизации
	min_n: int = 1024  # диапазон длины сигнала
	max_n: int = 2048
	bw_frac_min: float = 0.05  # диапазон относительной полосы
	bw_frac_max: float = 0.30
	freq_offset_frac: float = 0.06  # максимальный относительный частотный сдвиг
	polytime_phase_states: int = 2  # число фазовых состояний для polytime-сигналов
	multipath: bool = False  # включить многолучёвость
	phase_noise: bool = False  # включить фазовый шум
	iq_imbalance: bool = False  # включить дисбаланс I/Q
	amplitude_jitter: bool = False  # включить амплитудный джиттер


# Переводит значение из дБ в линейный масштаб
def db_to_linear(db: float) -> float:
	return 10.0 ** (db / 10.0)


# Нормирует комплексный IQ-сигнал к единичной средней мощности
def normalize_power(x: NDArray[np.complex128]) -> NDArray[np.complex128]:
	p = float(np.mean(np.abs(x) ** 2))
	if p <= 0.0 or not np.isfinite(p):
		return x.astype(np.complex128)
	return (x / math.sqrt(p)).astype(np.complex128)


# Возвращает случайное целое число в замкнутом диапазоне [low, high]
def random_int(rng: np.random.Generator, low: int, high: int) -> int:
	return int(rng.integers(low, high + 1))


# Случайно выбирает полосу сигнала как долю от частоты дискретизации
def choose_bandwidth(fs: float, cfg: GeneratorConfig, rng: np.random.Generator) -> float:
	return float(fs * rng.uniform(cfg.bw_frac_min, cfg.bw_frac_max))


# Добавляет комплексный AWGN к сигналу при заданном отношении сигнал-шум
def add_awgn(x: NDArray[np.complex128], snr_db: float, rng: np.random.Generator) -> NDArray[np.complex128]:
	signal_power = float(np.mean(np.abs(x) ** 2))
	if signal_power <= 0.0:
		signal_power = 1.0
	noise_power = signal_power / db_to_linear(snr_db)
	sigma = math.sqrt(noise_power / 2.0)
	noise = sigma * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
	return (x + noise).astype(np.complex128)


# Добавляет случайный частотный сдвиг и начальную фазу к IQ-сигналу
def apply_frequency_offset(x: NDArray[np.complex128], fs: float, cfg: GeneratorConfig, rng: np.random.Generator):
	n = x.size
	t = np.arange(n, dtype=np.float64) / fs
	f0 = float(rng.uniform(-cfg.freq_offset_frac * fs, cfg.freq_offset_frac * fs))
	phi0 = float(rng.uniform(0.0, 2.0 * np.pi))
	y = x * np.exp(1j * (2.0 * np.pi * f0 * t + phi0))
	return y.astype(np.complex128), f0, phi0


# Добавляет простую многолучёвость: задержанную и ослабленную копию сигнала
def apply_simple_multipath(x: NDArray[np.complex128], fs: float, rng: np.random.Generator):
	delay_s = float(rng.uniform(20e-9, 1.0e-6))
	delay = max(1, int(round(delay_s * fs)))
	gain_db = float(rng.uniform(-20.0, -6.0))
	gain = 10.0 ** (gain_db / 20.0)
	phase = float(rng.uniform(0.0, 2.0 * np.pi))
	echo = np.zeros_like(x)
	if delay < x.size:
		echo[delay:] = x[:-delay]
	y = x + gain * np.exp(1j * phase) * echo
	return normalize_power(y), {"path_delay_samples": delay, "path_gain_db": gain_db}


# Моделирует фазовый шум как случайное блуждание фазы
def apply_phase_noise(x: NDArray[np.complex128], rng: np.random.Generator):
	step_std = float(rng.uniform(1e-4, 2e-3))
	phase = np.cumsum(rng.normal(0.0, step_std, size=x.size))
	return (x * np.exp(1j * phase)).astype(np.complex128), {"phase_noise_step_std": step_std}


# Моделирует дисбаланс усиления и фазовую ошибку между I- и Q-каналами
def apply_iq_imbalance(x: NDArray[np.complex128], rng: np.random.Generator):
	gain_i = float(rng.uniform(0.92, 1.08))
	gain_q = float(rng.uniform(0.92, 1.08))
	phase_err = float(rng.uniform(-5.0, 5.0) * np.pi / 180.0)
	i = np.real(x) * gain_i
	q = np.imag(x) * gain_q
	y = i + 1j * (q * np.cos(phase_err) + i * np.sin(phase_err))
	return normalize_power(y.astype(np.complex128)), {
		"iq_gain_i": gain_i,
		"iq_gain_q": gain_q,
		"iq_phase_error_rad": phase_err,
	}


# Моделирует дисбаланс усиления и фазовую ошибку между I- и Q-каналами
def apply_amplitude_jitter(x: NDArray[np.complex128], fs: float, rng: np.random.Generator):
	n = x.size
	t = np.arange(n, dtype=np.float64) / fs
	rate = float(rng.uniform(0.02, 0.20) * fs / n)
	depth = float(rng.uniform(0.02, 0.10))
	phi = float(rng.uniform(0.0, 2.0 * np.pi))
	env = 1.0 + depth * np.sin(2.0 * np.pi * rate * t + phi)
	return normalize_power((x * env).astype(np.complex128)), {"amplitude_jitter_depth": depth}


# Преобразует список значений в кусочно-постоянный профиль длиной n
def piecewise_constant(values: Sequence[float], n: int) -> NDArray[np.float64]:
	values_arr = np.asarray(values, dtype=np.float64)
	edges = np.linspace(0, n, len(values_arr) + 1, dtype=int)
	out = np.empty(n, dtype=np.float64)
	for i, value in enumerate(values_arr):
		out[edges[i]:edges[i + 1]] = value
	return out


# Создаёт комплексный фазокодированный сигнал по последовательности фаз
def phases_to_waveform(phases: Sequence[float], n: int) -> NDArray[np.complex128]:
	phi = piecewise_constant(phases, n)
	return normalize_power(np.exp(1j * phi).astype(np.complex128))


# Создаёт комплексный сигнал по заданному профилю мгновенной частоты
def freq_to_waveform(f_inst_hz: NDArray[np.float64], fs: float) -> NDArray[np.complex128]:
	phase = 2.0 * np.pi * np.cumsum(f_inst_hz, dtype=np.float64) / fs
	return normalize_power(np.exp(1j * phase).astype(np.complex128))


# Генерирует прямоугольный сигнал с постоянной комплексной огибающей
def make_rect(n: int):
	x = np.ones(n, dtype=np.complex128)
	return normalize_power(x), {"modulation_model": "rectangular_constant_envelope"}


# Генерирует немодулированный CW-сигнал со случайной частотой
def make_unmodulated_cw(n: int, fs: float, rng: np.random.Generator):
	t = np.arange(n, dtype=np.float64) / fs
	f = float(rng.uniform(-0.04 * fs, 0.04 * fs))
	x = np.exp(1j * 2.0 * np.pi * f * t)
	return normalize_power(x.astype(np.complex128)), {"cw_offset_hz": f}


# Генерирует ЛЧМ-сигнал с линейным изменением мгновенной частоты
def make_lfm(n: int, fs: float, cfg: GeneratorConfig, rng: np.random.Generator):
	b = choose_bandwidth(fs, cfg, rng)
	t = np.arange(n, dtype=np.float64) / fs
	duration = n / fs
	f_inst = -b / 2.0 + b * t / duration
	return freq_to_waveform(f_inst, fs), {"bandwidth_hz": b, "sweep": "linear_centered"}


# Генерирует НЧМ-сигнал с нелинейным законом изменения частоты
def make_nlfm(n: int, fs: float, cfg: GeneratorConfig, rng: np.random.Generator):
	b = choose_bandwidth(fs, cfg, rng)
	u = np.linspace(-1.0, 1.0, n, dtype=np.float64)
	power = float(rng.uniform(1.5, 3.5))
	curve = np.sign(u) * np.abs(u) ** power
	sinus = float(rng.uniform(0.05, 0.20)) * np.sin(2.0 * np.pi * rng.uniform(1.0, 3.0) * (u + 1.0) / 2.0)
	curve = curve + sinus
	curve = curve - np.mean(curve)
	curve = curve / max(np.max(np.abs(curve)), 1e-8)
	f_inst = (b / 2.0) * curve
	return freq_to_waveform(f_inst, fs), {"bandwidth_hz": b, "nlfm_power": power}


# Генерирует FMCW-сигнал с восходящим, нисходящим или треугольным законом перестройки частоты
def make_fmcw(kind: str, n: int, fs: float, cfg: GeneratorConfig, rng: np.random.Generator):
	b = choose_bandwidth(fs, cfg, rng)
	cycles = int(rng.choice([2, 3, 4, 5]))
	u = (np.arange(n, dtype=np.float64) / n * cycles) % 1.0
	if kind == "FMCW_Up":
		shape = -1.0 + 2.0 * u
	elif kind == "FMCW_Down":
		shape = 1.0 - 2.0 * u
	elif kind == "FMCW_Triangular":
		shape = 2.0 * np.abs(2.0 * u - 1.0) - 1.0
	else:
		raise ValueError(kind)
	f_inst = (b / 2.0) * shape
	return freq_to_waveform(f_inst, fs), {"bandwidth_hz": b, "fmcw_cycles": cycles, "sweep": kind}


# Генерирует бинарный фазокодированный сигнал на основе кода Баркера
def make_barker(n: int, rng: np.random.Generator):
	length = int(rng.choice([7, 11, 13]))
	code = np.asarray(BARKER_CODES[length], dtype=np.float64)
	phases = np.where(code > 0.0, 0.0, np.pi)
	return phases_to_waveform(phases, n), {"code_length": length, "phase_levels": [0.0, float(np.pi)]}


# Генерирует максимальную LFSR-последовательность заданного порядка
def lfsr_sequence(order: int) -> NDArray[np.int8]:
	taps = MLS_TAPS[order]
	state = np.ones(order, dtype=np.int8)
	out = []
	for _ in range(2 ** order - 1):
		out.append(int(state[-1]))
		fb = 0
		for tap in taps:
			fb ^= int(state[-tap])
		state[1:] = state[:-1]
		state[0] = fb
	return np.asarray(out, dtype=np.int8)


# Генерирует бинарный фазокодированный сигнал на основе MLS-кода
def make_mls(n: int, rng: np.random.Generator):
	order = int(rng.choice([5, 6, 7, 8]))
	seq = lfsr_sequence(order)
	phases = np.where(seq > 0, 0.0, np.pi)
	return phases_to_waveform(phases, n), {"mls_order": order, "code_length": int(len(seq))}


# Строит бинарную последовательность Лежандра по квадратичным вычетам
def legendre_sequence(p: int) -> NDArray[np.int8]:
	residues = {(k * k) % p for k in range(1, p)}
	seq = np.empty(p, dtype=np.int8)
	for n in range(p):
		if n == 0:
			seq[n] = 1
		elif n in residues:
			seq[n] = 1
		else:
			seq[n] = -1
	return seq


# Генерирует фазокодированный сигнал на основе последовательности Лежандра
def make_legendre(n: int, rng: np.random.Generator):
	p = int(rng.choice([31, 47, 59, 61]))
	seq = legendre_sequence(p)
	phases = np.where(seq > 0, 0.0, np.pi)
	return phases_to_waveform(phases, n), {"legendre_prime": p, "code_length": p}


# Проверяет условие уникальности разностных векторов для Costas-перестановки
def is_costas_permutation(p: Sequence[int]) -> bool:
	seen = set()
	m = len(p)
	for i in range(m):
		for j in range(i + 1, m):
			d = (j - i, p[j] - p[i])
			if d in seen:
				return False
			seen.add(d)
	return True


# Возвращает все Costas-перестановки длины m с кешированием результата
@lru_cache(maxsize=None)
def all_costas_permutations(m: int) -> Tuple[Tuple[int, ...], ...]:
	return tuple(p for p in permutations(range(m)) if is_costas_permutation(p))


# Выбирает случайную Costas-последовательность для частотных скачков
def random_costas_sequence(m: int, rng: np.random.Generator) -> NDArray[np.int64]:
	candidates = all_costas_permutations(m)
	if not candidates:
		return rng.permutation(m).astype(np.int64)
	return np.asarray(candidates[int(rng.integers(0, len(candidates)))], dtype=np.int64)


# Генерирует частотно-перестраиваемый сигнал Costas
def make_costas(n: int, fs: float, cfg: GeneratorConfig, rng: np.random.Generator):
	m = int(rng.choice([4, 5, 6, 7]))
	b = choose_bandwidth(fs, cfg, rng)
	perm = random_costas_sequence(m, rng)
	freq_grid = np.linspace(-b / 2.0, b / 2.0, m)
	hop_freqs = freq_grid[perm]
	f_inst = piecewise_constant(hop_freqs, n)
	return freq_to_waveform(f_inst, fs), {"num_hops": m, "bandwidth_hz": b, "costas_sequence": perm.tolist()}


# Генерирует сигнал FH-FSK со случайной последовательностью частотных скачков
def make_fh_fsk(n: int, fs: float, cfg: GeneratorConfig, rng: np.random.Generator):
	b = choose_bandwidth(fs, cfg, rng)
	tones = int(rng.choice([4, 8, 16]))
	hops = int(rng.choice([8, 12, 16, 24, 32]))
	freq_grid = np.linspace(-b / 2.0, b / 2.0, tones)
	symbols = rng.integers(0, tones, size=hops)
	f_inst = piecewise_constant(freq_grid[symbols], n)
	return freq_to_waveform(f_inst, fs), {"tones": tones, "num_hops": hops, "bandwidth_hz": b}


# Генерирует SFCW-сигнал со ступенчатой перестройкой частоты
def make_sfcw(n: int, fs: float, cfg: GeneratorConfig, rng: np.random.Generator):
	b = choose_bandwidth(fs, cfg, rng)
	steps = int(rng.choice([16, 24, 32, 48]))
	repeats = int(rng.choice([1, 2, 3]))
	seq = np.tile(np.linspace(-b / 2.0, b / 2.0, steps), repeats)
	f_inst = piecewise_constant(seq, n)
	return freq_to_waveform(f_inst, fs), {"num_steps": steps, "repeats": repeats, "bandwidth_hz": b}


# Формирует последовательность фаз кода Франка
def frank_phases(m: int) -> NDArray[np.float64]:
	return np.asarray([2.0 * np.pi * r * c / m for r in range(m) for c in range(m)], dtype=np.float64)


# Формирует последовательность фаз P1-кода
def p1_phases(m: int) -> NDArray[np.float64]:
	phases = []
	center = (m - 1.0) / 2.0
	for r in range(m):
		for c in range(m):
			idx = r * m + c
			phases.append((-2.0 * np.pi / m) * (center - r) * idx)
	return np.asarray(phases, dtype=np.float64)


# Формирует последовательность фаз P2-кода
def p2_phases(m: int) -> NDArray[np.float64]:
	phases = []
	center = (m - 1.0) / 2.0
	for r in range(m):
		for c in range(m):
			phases.append((2.0 * np.pi / m) * (center - r) * (center - c))
	return np.asarray(phases, dtype=np.float64)


# Формирует квадратичный фазовый закон P3-кода
def p3_phases(num_chips: int) -> NDArray[np.float64]:
	i = np.arange(num_chips, dtype=np.float64)
	return (np.pi / num_chips) * i ** 2


# Формирует квадратичный фазовый закон P4-кода
def p4_phases(num_chips: int) -> NDArray[np.float64]:
	i = np.arange(num_chips, dtype=np.float64)
	return (np.pi / num_chips) * i * (i - num_chips)


# Генерирует полифазный код Frank/P1/P2/P3/P4
def make_polyphase(kind: str, n: int, rng: np.random.Generator):
	if kind == "Frank":
		m = int(rng.choice([6, 7, 8]))
		phases = frank_phases(m)
		params = {"code_length": int(m * m), "M": m}
	elif kind == "P1":
		m = int(rng.choice([6, 7, 8]))
		phases = p1_phases(m)
		params = {"code_length": int(m * m), "M": m}
	elif kind == "P2":
		m = int(rng.choice([6, 7, 8]))
		phases = p2_phases(m)
		params = {"code_length": int(m * m), "M": m}
	elif kind == "P3":
		chips = int(rng.choice([36, 39, 64]))
		phases = p3_phases(chips)
		params = {"code_length": chips}
	elif kind == "P4":
		chips = int(rng.choice([36, 39, 64]))
		phases = p4_phases(chips)
		params = {"code_length": chips}
	else:
		raise ValueError(kind)
	return phases_to_waveform(np.mod(phases, 2.0 * np.pi), n), {**params, "phase_code": kind}


# Квантует фазовый закон на заданное число фазовых состояний
def quantize_phase(phi: NDArray[np.float64], n_states: int) -> NDArray[np.float64]:
	step = 2.0 * np.pi / max(2, n_states)
	wrapped = np.mod(phi, 2.0 * np.pi)
	levels = np.floor((wrapped + step / 2.0) / step).astype(int) % max(2, n_states)
	return levels.astype(np.float64) * step


# Формирует ступенчатый профиль мгновенной частоты для polytime-сигналов
def stepped_frequency_profile(n: int, b: float, groups: int, mode: str) -> NDArray[np.float64]:
	edges = np.linspace(0, n, groups + 1, dtype=int)
	f = np.zeros(n, dtype=np.float64)
	if mode == "leading_zero":
		steps = np.linspace(0.0, b, groups)
	elif mode == "centered":
		steps = np.linspace(-b / 2.0, b / 2.0, groups)
	else:
		raise ValueError(mode)
	for g, value in enumerate(steps):
		f[edges[g]:edges[g + 1]] = value
	if mode == "leading_zero":
		f = f - np.mean(f) + b / 8.0
	return f


# Генерирует polytime-сигналы T1–T4 с квантованием фазового закона
def make_polytime(kind: str, n: int, fs: float, cfg: GeneratorConfig, rng: np.random.Generator):
	b = choose_bandwidth(fs, cfg, rng)
	groups = int(rng.choice([4, 5, 6]))
	t = np.arange(n, dtype=np.float64) / fs
	duration = n / fs
	states = cfg.polytime_phase_states
	if kind == "T1":
		f_inst = stepped_frequency_profile(n, b, groups, mode="leading_zero")
		model = "quantized_stepped_rf_leading_zero"
	elif kind == "T2":
		f_inst = stepped_frequency_profile(n, b, groups, mode="centered")
		model = "quantized_stepped_rf_centered"
	elif kind == "T3":
		f_inst = b * t / duration
		f_inst = f_inst - np.mean(f_inst) + b / 8.0
		model = "quantized_lfm_leading_zero"
	elif kind == "T4":
		f_inst = -b / 2.0 + b * t / duration
		model = "quantized_lfm_centered"
	else:
		raise ValueError(kind)
	phi_cont = 2.0 * np.pi * np.cumsum(f_inst) / fs
	phi_q = quantize_phase(phi_cont, states)
	x = np.exp(1j * phi_q).astype(np.complex128)
	return normalize_power(x), {"bandwidth_hz": b, "num_groups": groups, "phase_states": states,
	                            "polytime_model": model}


# Формирует нормированное квадратное QAM-созвездие заданного порядка
def qam_constellation(order: int) -> NDArray[np.complex128]:
	side = int(round(math.sqrt(order)))
	vals = np.arange(-(side - 1), side, 2, dtype=np.float64)
	points = np.asarray([i + 1j * q for i in vals for q in vals], dtype=np.complex128)
	return normalize_power(points)


# Строит RRC-фильтр для импульсного формирования цифровых сигналов
def rrc_filter(beta: float, span: int, sps: int) -> NDArray[np.float64]:
	n = span * sps
	t = (np.arange(-n / 2, n / 2 + 1, dtype=np.float64) / sps)
	h = np.zeros_like(t)
	for idx, ti in enumerate(t):
		if abs(ti) < 1e-12:
			h[idx] = 1.0 - beta + 4.0 * beta / np.pi
		elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-10:
			h[idx] = (beta / math.sqrt(2.0)) * (
					(1.0 + 2.0 / np.pi) * math.sin(np.pi / (4.0 * beta)) + (1.0 - 2.0 / np.pi) * math.cos(
				np.pi / (4.0 * beta)))
		else:
			num = math.sin(np.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * math.cos(np.pi * ti * (1.0 + beta))
			den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
			h[idx] = num / den
	h = h / math.sqrt(np.sum(h ** 2))
	return h.astype(np.float64)


# Выполняет upsampling и RRC-фильтрацию последовательности символов
def pulse_shape(symbols: NDArray[np.complex128], sps: int, n: int, rng: np.random.Generator) -> NDArray[np.complex128]:
	up = np.zeros(len(symbols) * sps, dtype=np.complex128)
	up[::sps] = symbols
	beta = float(rng.uniform(0.20, 0.50))
	taps = rrc_filter(beta=beta, span=8, sps=sps)
	y = lfilter(taps, [1.0], up)
	if y.size < n:
		reps = int(np.ceil(n / y.size))
		y = np.tile(y, reps)
	return normalize_power(y[:n].astype(np.complex128))


# Генерирует цифровой PSK-сигнал с RRC-импульсным формированием
def make_psk_comm(kind: str, n: int, rng: np.random.Generator):
	if kind == "BPSK_Random":
		m = 2
	elif kind == "QPSK":
		m = 4
	elif kind == "8PSK":
		m = 8
	else:
		raise ValueError(kind)
	sps = int(rng.choice([6, 8, 10, 12, 16]))
	nsym = int(np.ceil(n / sps) + 16)
	idx = rng.integers(0, m, size=nsym)
	phase_offset = np.pi / m if m > 2 else 0.0
	symbols = np.exp(1j * (2.0 * np.pi * idx / m + phase_offset)).astype(np.complex128)
	return pulse_shape(symbols, sps, n, rng), {"mod_order": m, "sps": sps, "pulse_shape": "rrc"}


# Генерирует цифровой QAM-сигнал с RRC-импульсным формированием
def make_qam_comm(kind: str, n: int, rng: np.random.Generator):
	order = 16 if kind == "16QAM" else 64
	constellation = qam_constellation(order)
	sps = int(rng.choice([6, 8, 10, 12, 16]))
	nsym = int(np.ceil(n / sps) + 16)
	symbols = constellation[rng.integers(0, len(constellation), size=nsym)]
	return pulse_shape(symbols, sps, n, rng), {"mod_order": order, "sps": sps, "pulse_shape": "rrc"}


# Генерирует PAM4-сигнал с четырьмя амплитудными уровнями
def make_pam4(n: int, rng: np.random.Generator):
	levels = np.asarray([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
	levels = levels / math.sqrt(np.mean(levels ** 2))
	sps = int(rng.choice([6, 8, 10, 12, 16]))
	nsym = int(np.ceil(n / sps) + 16)
	symbols = levels[rng.integers(0, 4, size=nsym)].astype(np.complex128)
	return pulse_shape(symbols, sps, n, rng), {"mod_order": 4, "sps": sps, "pulse_shape": "rrc"}


# Строит гауссов фильтр для сглаживания символов в GFSK
def gaussian_filter_1d(bt: float, sps: int, span: int = 4) -> NDArray[np.float64]:
	t = np.arange(-span * sps, span * sps + 1, dtype=np.float64) / sps
	alpha = math.sqrt(math.log(2.0)) / max(bt, 1e-6)
	h = np.exp(-(alpha * t) ** 2)
	h = h / np.sum(h)
	return h.astype(np.float64)


# Генерирует GFSK или CPFSK через частотное отклонение и интегрирование фазы
def make_fsk_comm(kind: str, n: int, fs: float, rng: np.random.Generator):
	sps = int(rng.choice([8, 10, 12, 16, 20]))
	nsym = int(np.ceil(n / sps) + 8)
	bits = rng.choice([-1.0, 1.0], size=nsym)
	symbols = np.repeat(bits, sps)
	if kind == "GFSK":
		bt = float(rng.uniform(0.3, 0.7))
		symbols = np.convolve(symbols, gaussian_filter_1d(bt, sps), mode="same")
	elif kind != "CPFSK":
		raise ValueError(kind)
	symbols = symbols[:n]
	h = float(rng.uniform(0.35, 0.75))
	f_dev = h * fs / (2.0 * sps)
	f_inst = f_dev * symbols
	return freq_to_waveform(f_inst, fs), {"sps": sps, "mod_index": h, "fsk_model": kind}


# Генерирует OFDM-подобный радарный сигнал с активными поднесущими и циклическим префиксом
def make_ofdm_radar(n: int, rng: np.random.Generator):
	nfft = int(rng.choice([64, 96, 128]))
	active = int(rng.choice([24, 32, 48, 64]))
	active = min(active, nfft - 4)
	cp = max(4, nfft // 8)
	nsym = int(np.ceil(n / (nfft + cp)) + 2)
	blocks = []
	center = nfft // 2
	half = active // 2
	carriers = np.arange(center - half, center + half, dtype=int)
	for _ in range(nsym):
		freq = np.zeros(nfft, dtype=np.complex128)
		qpsk = np.exp(1j * (np.pi / 4.0 + np.pi / 2.0 * rng.integers(0, 4, size=carriers.size)))
		freq[carriers] = qpsk
		time_sym = np.fft.ifft(np.fft.ifftshift(freq))
		blocks.append(np.concatenate([time_sym[-cp:], time_sym]))
	y = np.concatenate(blocks)
	if y.size < n:
		y = np.tile(y, int(np.ceil(n / y.size)))
	return normalize_power(y[:n].astype(np.complex128)), {"ofdm_nfft": nfft, "active_subcarriers": active, "cp": cp}


# Создаёт низкочастотное сообщение как сумму случайных тонов
def audio_like_message(n: int, fs: float, rng: np.random.Generator) -> NDArray[np.float64]:
	t = np.arange(n, dtype=np.float64) / fs
	# Use normalized low-rate tones relative to the observation time, not audio Hz.
	msg = np.zeros(n, dtype=np.float64)
	num_tones = int(rng.choice([2, 3, 4]))
	duration = n / fs
	for _ in range(num_tones):
		cycles = float(rng.uniform(1.0, 8.0))
		amp = float(rng.uniform(0.3, 1.0))
		phase = float(rng.uniform(0.0, 2.0 * np.pi))
		msg += amp * np.sin(2.0 * np.pi * cycles * t / duration + phase)
	msg = msg - np.mean(msg)
	msg = msg / max(np.max(np.abs(msg)), 1e-8)
	return msg


# Генерирует аналоговый FM-сигнал по низкочастотному сообщению
def make_bfm(n: int, fs: float, rng: np.random.Generator):
	msg = audio_like_message(n, fs, rng)
	f_dev = float(rng.uniform(0.01, 0.08) * fs)
	f_inst = f_dev * msg
	return freq_to_waveform(f_inst, fs), {"fm_deviation_hz": f_dev}


# Генерирует DSB-AM-сигнал с остаточной несущей
def make_dsb_am(n: int, fs: float, rng: np.random.Generator):
	msg = audio_like_message(n, fs, rng)
	mu = float(rng.uniform(0.3, 0.9))
	x = (1.0 + mu * msg).astype(np.complex128)
	return normalize_power(x), {"am_index": mu, "am_type": "DSB_AM"}


# Генерирует SSB-AM-сигнал через аналитическое представление сообщения
def make_ssb_am(n: int, fs: float, rng: np.random.Generator):
	msg = audio_like_message(n, fs, rng)
	analytic = hilbert(msg).astype(np.complex128)
	residual = float(rng.uniform(0.0, 0.15))
	x = analytic + residual
	return normalize_power(x), {"am_type": "SSB_AM", "residual_carrier": residual}


# Генерирует чистый комплексный шум как отдельный класс
def make_noise_only(n: int, rng: np.random.Generator):
	x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
	return normalize_power(x.astype(np.complex128)), {"modulation_model": "noise_only_pre_awgn"}


# Главная функция генерации одного IQ-сигнала заданного класса с опциональными искажениями
def make_waveform(kind: str, n: int, fs: float, cfg: GeneratorConfig, rng: np.random.Generator):
	if kind == "Rect":
		x, params = make_rect(n)
	elif kind == "Unmodulated_CW":
		x, params = make_unmodulated_cw(n, fs, rng)
	elif kind == "NoiseOnly":
		x, params = make_noise_only(n, rng)
	elif kind == "LFM":
		x, params = make_lfm(n, fs, cfg, rng)
	elif kind == "NLFM":
		x, params = make_nlfm(n, fs, cfg, rng)
	elif kind in {"FMCW_Up", "FMCW_Down", "FMCW_Triangular"}:
		x, params = make_fmcw(kind, n, fs, cfg, rng)
	elif kind == "Barker":
		x, params = make_barker(n, rng)
	elif kind == "MLS":
		x, params = make_mls(n, rng)
	elif kind == "Legendre":
		x, params = make_legendre(n, rng)
	elif kind == "Costas":
		x, params = make_costas(n, fs, cfg, rng)
	elif kind == "FH_FSK":
		x, params = make_fh_fsk(n, fs, cfg, rng)
	elif kind == "SFCW":
		x, params = make_sfcw(n, fs, cfg, rng)
	elif kind in {"Frank", "P1", "P2", "P3", "P4"}:
		x, params = make_polyphase(kind, n, rng)
	elif kind in {"T1", "T2", "T3", "T4"}:
		x, params = make_polytime(kind, n, fs, cfg, rng)
	elif kind in {"BPSK_Random", "QPSK", "8PSK"}:
		x, params = make_psk_comm(kind, n, rng)
	elif kind in {"16QAM", "64QAM"}:
		x, params = make_qam_comm(kind, n, rng)
	elif kind == "PAM4":
		x, params = make_pam4(n, rng)
	elif kind in {"GFSK", "CPFSK"}:
		x, params = make_fsk_comm(kind, n, fs, rng)
	elif kind == "OFDM_Radar":
		x, params = make_ofdm_radar(n, rng)
	elif kind == "B_FM":
		x, params = make_bfm(n, fs, rng)
	elif kind == "DSB_AM":
		x, params = make_dsb_am(n, fs, rng)
	elif kind == "SSB_AM":
		x, params = make_ssb_am(n, fs, rng)
	else:
		raise ValueError(f"Unsupported waveform class: {kind}")

	if kind != "NoiseOnly":
		x, f0, phi0 = apply_frequency_offset(x, fs, cfg, rng)
		params.update({"freq_offset_hz": f0, "initial_phase_rad": phi0})

	if cfg.amplitude_jitter and kind != "NoiseOnly":
		x, extra = apply_amplitude_jitter(x, fs, rng)
		params.update(extra)
	if cfg.iq_imbalance:
		x, extra = apply_iq_imbalance(x, rng)
		params.update(extra)
	if cfg.phase_noise and kind != "NoiseOnly":
		x, extra = apply_phase_noise(x, rng)
		params.update(extra)
	if cfg.multipath and kind != "NoiseOnly":
		x, extra = apply_simple_multipath(x, fs, rng)
		params.update(extra)
	return normalize_power(x), params


# Строит изображение STFT-спектрограммы в оттенках серого
def spectrogram_image(x: NDArray[np.complex128], fs: float, cfg: StftConfig) -> Image.Image:
	nperseg = min(cfg.nperseg, x.size)
	noverlap = min(cfg.noverlap, max(0, nperseg - 1))
	nfft = max(cfg.nfft, nperseg)
	_, _, z = stft(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft, return_onesided=False,
	               boundary=None, padded=False)
	s = np.fft.fftshift(np.abs(z) ** 2, axes=0)
	s_db = 10.0 * np.log10(s + 1e-14)
	s_db -= np.max(s_db)
	s_db = np.clip(s_db, -cfg.dynamic_range_db, 0.0)
	img = ((s_db + cfg.dynamic_range_db) / cfg.dynamic_range_db * 255.0).astype(np.uint8)
	img = np.flipud(img)
	return Image.fromarray(img, mode="L").resize((cfg.image_size, cfg.image_size), Image.Resampling.BICUBIC)


# Строит практическое псевдо-CWD изображение с экспоненциальным ядром сглаживания
def cwd_image(x: NDArray[np.complex128], fs: float, cfg: CwdConfig) -> Image.Image:
	if cfg.sigma <= 0.0:
		raise ValueError("cwd sigma must be positive")
	x = normalize_power(np.asarray(x, dtype=np.complex128))
	if cfg.max_samples > 0 and x.size > cfg.max_samples:
		x = normalize_power(resample(x, cfg.max_samples).astype(np.complex128))
	n = int(x.size)
	max_lag = min(int(cfg.max_lag), max(1, (n - 1) // 2))
	lags = np.arange(-max_lag, max_lag + 1, dtype=np.int64)
	r = np.zeros((n, lags.size), dtype=np.complex128)
	for col, lag in enumerate(lags):
		a = int(abs(lag))
		if a == 0:
			r[:, col] = x * np.conj(x)
		elif lag > 0:
			r[a:n - a, col] = x[2 * a:n] * np.conj(x[:n - 2 * a])
		else:
			r[a:n - a, col] = x[:n - 2 * a] * np.conj(x[2 * a:n])
	ambiguity = np.fft.fft(r, axis=0)
	xi = 2.0 * np.pi * np.fft.fftfreq(n)
	tau = lags.astype(np.float64)
	kernel = np.exp(-((xi[:, None] * tau[None, :]) ** 2) / cfg.sigma)
	smoothed = np.fft.ifft(ambiguity * kernel, axis=0)
	nfft = max(int(cfg.nfft), lags.size)
	smoothed_zero_lag_first = np.fft.ifftshift(smoothed, axes=1)
	tfr = np.fft.fftshift(np.fft.fft(smoothed_zero_lag_first, n=nfft, axis=1), axes=1)
	power = np.abs(tfr).T
	power_db = 10.0 * np.log10(power + 1e-14)
	power_db -= np.max(power_db)
	power_db = np.clip(power_db, -cfg.dynamic_range_db, 0.0)
	img = ((power_db + cfg.dynamic_range_db) / cfg.dynamic_range_db * 255.0).astype(np.uint8)
	img = np.flipud(img)
	return Image.fromarray(img, mode="L").resize((cfg.image_size, cfg.image_size), Image.Resampling.BICUBIC)


# Выбирает тип частотно-временного анализа: STFT или CWD
def time_frequency_image(x: NDArray[np.complex128], fs: float, tfa: str, stft_cfg: StftConfig,
                         cwd_cfg: CwdConfig) -> Image.Image:
	if tfa == "stft":
		return spectrogram_image(x, fs, stft_cfg)
	if tfa == "cwd":
		return cwd_image(x, fs, cwd_cfg)
	raise ValueError(tfa)


# Случайно назначает пример в train/val/test по заданным долям
def split_name(rng: np.random.Generator, train: float, val: float) -> str:
	u = float(rng.random())
	if u < train:
		return "train"
	if u < train + val:
		return "val"
	return "test"


# Детерминированно делит каждую группу класс/SNR на train, val и test
def stratified_split_name(sample_idx: int, samples_per_group: int, train: float, val: float) -> str:
	# Deterministic split inside every class/SNR group. This prevents missing
	# classes in train/val/test when experiments use small sample counts.
	train_n = int(round(samples_per_group * train))
	val_n = int(round(samples_per_group * val))
	train_n = min(max(train_n, 1 if samples_per_group >= 3 else 0), samples_per_group)
	val_n = min(max(val_n, 1 if samples_per_group >= 3 else 0), max(0, samples_per_group - train_n))
	if sample_idx < train_n:
		return "train"
	if sample_idx < train_n + val_n:
		return "val"
	return "test"


# Преобразует значение ОСШ в безопасный фрагмент имени файла
def safe_snr_name(snr_db: float) -> str:
	if float(snr_db).is_integer():
		return f"{int(snr_db):+03d}dB".replace("+", "p").replace("-", "m")
	return f"{snr_db:+.1f}dB".replace("+", "p").replace("-", "m").replace(".", "p")


# Определяет итоговый список классов по --classes или --profile
def classes_from_args(args: argparse.Namespace) -> Tuple[str, ...]:
	if args.classes:
		classes = tuple(args.classes)
	else:
		classes = PROFILE_CLASSES[args.profile]
	unknown = sorted(set(classes) - set(MIXED_RF_CLASSES))
	if unknown:
		raise ValueError(f"Unknown classes: {unknown}")
	return classes


# Генерирует полный датасет частотно-временных изображений и metadata.csv
def generate_dataset(args: argparse.Namespace) -> None:
	out_dir = Path(args.out)
	out_dir.mkdir(parents=True, exist_ok=True)
	rng = np.random.default_rng(args.seed)
	gen_cfg = GeneratorConfig(
		fs=args.fs,
		min_n=args.min_n,
		max_n=args.max_n,
		bw_frac_min=args.bw_frac_min,
		bw_frac_max=args.bw_frac_max,
		freq_offset_frac=args.freq_offset_frac,
		polytime_phase_states=args.polytime_phase_states,
		multipath=args.multipath,
		phase_noise=args.phase_noise,
		iq_imbalance=args.iq_imbalance,
		amplitude_jitter=args.amplitude_jitter,
	)
	stft_cfg = StftConfig(args.stft_nperseg, args.stft_noverlap, args.stft_nfft, args.dynamic_range_db, args.image_size)
	cwd_cfg = CwdConfig(args.cwd_sigma, args.cwd_max_lag, args.cwd_nfft, args.cwd_max_samples, args.dynamic_range_db,
	                    args.image_size)
	classes = classes_from_args(args)
	if args.train_fraction + args.val_fraction >= 1.0:
		raise ValueError("train_fraction + val_fraction must be < 1.0")
	total = len(classes) * len(args.snrs) * args.samples_per_class_snr
	written = 0
	metadata_path = out_dir / "metadata.csv"
	config_path = out_dir / "generation_config.json"
	with config_path.open("w", encoding="utf-8") as f:
		json.dump({**vars(args), "classes_resolved": list(classes)}, f, ensure_ascii=False, indent=2)
	with metadata_path.open("w", newline="", encoding="utf-8") as f:
		fieldnames = [
			"relative_path", "split", "label", "domain", "family", "snr_db",
			"sample_index", "fs_hz", "num_samples", "tfa", "profile", "params_json",
		]
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for label in classes:
			info = CLASS_INFO[label]
			for snr_db in args.snrs:
				snr_tag = safe_snr_name(snr_db)
				for sample_idx in range(args.samples_per_class_snr):
					n = random_int(rng, args.min_n, args.max_n)
					x, params = make_waveform(label, n, args.fs, gen_cfg, rng)
					y = add_awgn(x, snr_db, rng)
					img = time_frequency_image(y, args.fs, args.tfa, stft_cfg, cwd_cfg)
					if args.split_mode == "stratified":
						split = stratified_split_name(sample_idx, args.samples_per_class_snr, args.train_fraction,
						                              args.val_fraction)
					else:
						split = split_name(rng, args.train_fraction, args.val_fraction)
					class_dir = out_dir / split / label
					class_dir.mkdir(parents=True, exist_ok=True)
					filename = f"{label}_snr{snr_tag}_{sample_idx:06d}.png"
					path = class_dir / filename
					img.save(path)
					rel = path.relative_to(out_dir).as_posix()
					writer.writerow({
						"relative_path": rel,
						"split": split,
						"label": label,
						"domain": info["domain"],
						"family": info["family"],
						"snr_db": snr_db,
						"sample_index": sample_idx,
						"fs_hz": args.fs,
						"num_samples": n,
						"tfa": args.tfa,
						"profile": args.profile,
						"params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
					})
					written += 1
					if args.progress and (written % args.progress == 0 or written == total):
						print(f"[{written:>7}/{total}] saved {rel}")
	print(f"Done. Images: {written}. Metadata: {metadata_path}")


# Создаёт обзорную картинку с одним частотно-временным портретом для каждого класса
def make_preview(args: argparse.Namespace) -> None:
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	out_dir = Path(args.out)
	out_dir.mkdir(parents=True, exist_ok=True)
	rng = np.random.default_rng(args.seed)
	gen_cfg = GeneratorConfig(
		fs=args.fs,
		min_n=args.min_n,
		max_n=args.max_n,
		bw_frac_min=args.bw_frac_min,
		bw_frac_max=args.bw_frac_max,
		freq_offset_frac=args.freq_offset_frac,
		polytime_phase_states=args.polytime_phase_states,
		multipath=args.multipath,
		phase_noise=args.phase_noise,
		iq_imbalance=args.iq_imbalance,
		amplitude_jitter=args.amplitude_jitter,
	)
	stft_cfg = StftConfig(args.stft_nperseg, args.stft_noverlap, args.stft_nfft, args.dynamic_range_db, args.image_size)
	cwd_cfg = CwdConfig(args.cwd_sigma, args.cwd_max_lag, args.cwd_nfft, args.cwd_max_samples, args.dynamic_range_db,
	                    args.image_size)
	classes = classes_from_args(args)
	cols = min(7, max(3, int(math.ceil(math.sqrt(len(classes))))))
	rows = int(math.ceil(len(classes) / cols))
	fig = plt.figure(figsize=(cols * 2.1, rows * 1.9))
	for idx, label in enumerate(classes, start=1):
		n = random_int(rng, args.min_n, args.max_n)
		x, _ = make_waveform(label, n, args.fs, gen_cfg, rng)
		y = add_awgn(x, args.preview_snr, rng)
		img = time_frequency_image(y, args.fs, args.tfa, stft_cfg, cwd_cfg)
		ax = fig.add_subplot(rows, cols, idx)
		ax.imshow(img, cmap="gray", aspect="auto")
		ax.set_title(label, fontsize=8)
		ax.axis("off")
	for idx in range(len(classes) + 1, rows * cols + 1):
		fig.add_subplot(rows, cols, idx).axis("off")
	fig.tight_layout()
	preview_path = out_dir / f"preview_{args.profile}_{args.tfa}.png"
	fig.savefig(preview_path, dpi=180)
	plt.close(fig)
	print(f"Preview saved: {preview_path}")


# Описывает параметры командной строки для генерации датасета
def build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Generate RF time-frequency image datasets.",
	                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument("--out", type=str, default="rf_dataset", help="Output dataset directory")
	p.add_argument("--profile", choices=sorted(PROFILE_CLASSES), default="mixed_rf", help="Class profile")
	p.add_argument("--classes", nargs="+", default=None, help="Optional explicit class subset")
	p.add_argument("--samples-per-class-snr", type=int, default=200, help="Images per class per SNR")
	p.add_argument("--snrs", nargs="+", type=float, default=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10],
	               help="SNR values in dB")
	p.add_argument("--seed", type=int, default=12345)
	p.add_argument("--fs", type=float, default=100e6)
	p.add_argument("--min-n", type=int, default=1024)
	p.add_argument("--max-n", type=int, default=2048)
	p.add_argument("--bw-frac-min", type=float, default=0.05)
	p.add_argument("--bw-frac-max", type=float, default=0.30)
	p.add_argument("--freq-offset-frac", type=float, default=0.06)
	p.add_argument("--polytime-phase-states", type=int, default=2)
	p.add_argument("--multipath", action="store_true")
	p.add_argument("--phase-noise", action="store_true")
	p.add_argument("--iq-imbalance", action="store_true")
	p.add_argument("--amplitude-jitter", action="store_true")
	p.add_argument("--image-size", type=int, default=224)
	p.add_argument("--tfa", choices=["stft", "cwd"], default="cwd")
	p.add_argument("--stft-nperseg", type=int, default=128)
	p.add_argument("--stft-noverlap", type=int, default=112)
	p.add_argument("--stft-nfft", type=int, default=256)
	p.add_argument("--dynamic-range-db", type=float, default=55.0)
	p.add_argument("--cwd-sigma", type=float, default=1.0)
	p.add_argument("--cwd-max-lag", type=int, default=96)
	p.add_argument("--cwd-nfft", type=int, default=256)
	p.add_argument("--cwd-max-samples", type=int, default=1024)
	p.add_argument("--train-fraction", type=float, default=0.70)
	p.add_argument("--val-fraction", type=float, default=0.15)
	p.add_argument("--split-mode", choices=["stratified", "random"], default="stratified",
	               help="Dataset split assignment mode")
	p.add_argument("--progress", type=int, default=500, help="Print every N images, 0 disables")
	p.add_argument("--preview", action="store_true")
	p.add_argument("--preview-snr", type=float, default=0.0)
	return p


# Точка входа: запускает генерацию датасета или создание preview
def main() -> None:
	args = build_argparser().parse_args()
	if args.preview:
		make_preview(args)
	else:
		generate_dataset(args)


if __name__ == "__main__":
	main()
