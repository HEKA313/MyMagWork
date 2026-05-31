#!/usr/bin/env python3
"""
Этот файл описывает архитектуру нейронной сети. В нём есть две реализации:
– baseline_b0 - EfficientNet-B0 со стандартным MBConv;
– improved_b0 - улучшенный вариант с параллельной веткой глубинной свертки и веткой SimAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# Описание одной стадии EfficientNet
@dataclass(frozen=True)
class MBConvConfig:
	expand_ratio: int  # коэффициент расширения каналов
	kernel: int  # размер ядра свёртки
	stride: int  # шаг первого блока группы
	input_channels: int  # число входных каналов
	out_channels: int  # число выходных каналов
	num_layers: int  # количество повторений блока

	# Конфигурация стадий EfficientNet-B0 для входного изображения 224x224
	B0_CONFIG = (
		MBConvConfig(1, 3, 1, 32, 16, 1),
		MBConvConfig(6, 3, 2, 16, 24, 2),
		MBConvConfig(6, 5, 2, 24, 40, 2),
		MBConvConfig(6, 3, 2, 40, 80, 3),
		MBConvConfig(6, 5, 1, 80, 112, 3),
		MBConvConfig(6, 5, 2, 112, 192, 4),
		MBConvConfig(6, 3, 1, 192, 320, 1),
	)


# Базовый свёрточный блок: свёртка, пакетная нормализация и опциональная SiLU-активация
class ConvBNAct(nn.Sequential):
	def __init__(
			self,
			in_channels: int,  # Входные каналы
			out_channels: int,  # Выходные каналы
			kernel_size: int,  # размер ядра
			stride: int = 1,  # шаг
			groups: int = 1,  # число групп
			activation: bool = True,  # добавлять ли SiLU
	) -> None:
		padding = (kernel_size - 1) // 2
		layers = [
			nn.Conv2d(
				in_channels,
				out_channels,
				kernel_size=kernel_size,
				stride=stride,
				padding=padding,
				groups=groups,
				bias=False,
			),
			nn.BatchNorm2d(out_channels),
		]
		if activation:
			layers.append(nn.SiLU(inplace=True))
		super().__init__(*layers)


# SE-блок: вычисляет веса каналов и масштабирует признаки по каналам
class SqueezeExcitation(nn.Module):

	def __init__(self, input_channels: int, squeeze_channels: int) -> None:
		super().__init__()
		self.avgpool = nn.AdaptiveAvgPool2d(1)  # сжимает каждую карту признаков до одного числа
		self.fc1 = nn.Conv2d(input_channels, squeeze_channels, kernel_size=1)  # уменьшает число каналов
		self.act = nn.SiLU(inplace=True)  # нелинейность
		self.fc2 = nn.Conv2d(squeeze_channels, input_channels, kernel_size=1)  # возвращает число каналов обратно

	# SE-блок: вычисляет веса каналов и масштабирует признаки по каналам
	def forward(self, x: Tensor) -> Tensor:
		scale = self.avgpool(x)
		scale = self.fc1(scale)
		scale = self.act(scale)
		scale = torch.sigmoid(self.fc2(scale))
		return x * scale


# SimAM-внимание: без обучаемых параметров выделяет важные элементы карты признаков
class SimAM(nn.Module):
	def __init__(self, lambda_value: float = 1e-4) -> None:
		super().__init__()
		self.lambda_value = float(lambda_value)
		self.activation = nn.Sigmoid()

	def forward(self, x: Tensor) -> Tensor:
		_, _, h, w = x.shape
		n = h * w - 1
		if n <= 0:
			return x
		d = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
		variance = d.sum(dim=(2, 3), keepdim=True) / float(n)
		y = d / (4.0 * (variance + self.lambda_value)) + 0.5
		return x * self.activation(y)


# Стохастическая глубина: случайно отключает остаточную ветвь при обучении
class StochasticDepth(nn.Module):
	def __init__(self, p: float) -> None:
		super().__init__()
		self.p = float(p)

	def forward(self, x: Tensor) -> Tensor:
		if not self.training or self.p <= 0.0:
			return x
		keep_prob = 1.0 - self.p
		shape = [x.shape[0]] + [1] * (x.ndim - 1)
		mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
		return x.div(keep_prob) * mask


# Стандартный MBConv: expansion -> depthwise conv -> SE -> projection -> residual
class BaselineMBConv(nn.Module):
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			kernel_size: int,
			stride: int,
			expand_ratio: int,
			stochastic_depth_prob: float,
	) -> None:
		super().__init__()
		self.use_residual = stride == 1 and in_channels == out_channels
		hidden_dim = in_channels * expand_ratio
		layers = []
		if expand_ratio != 1:
			layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1))
		layers.append(
			ConvBNAct(
				hidden_dim,
				hidden_dim,
				kernel_size=kernel_size,
				stride=stride,
				groups=hidden_dim,
			)
		)
		squeeze_channels = max(1, in_channels // 4)
		layers.append(SqueezeExcitation(hidden_dim, squeeze_channels))
		layers.append(ConvBNAct(hidden_dim, out_channels, kernel_size=1, activation=False))
		self.block = nn.Sequential(*layers)
		self.stochastic_depth = StochasticDepth(stochastic_depth_prob)

	def forward(self, x: Tensor) -> Tensor:
		y = self.block(x)
		if self.use_residual:
			y = self.stochastic_depth(y) + x
		return y


# Улучшенный MBConv: параллельно извлекает локальные признаки и признаки внимания, затем объединяет их
class ImprovedMBConv(nn.Module):
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			kernel_size: int,
			stride: int,
			expand_ratio: int,
			stochastic_depth_prob: float,
			simam_lambda: float = 1e-4,
			block_dropout: float = 0.0,
	) -> None:
		super().__init__()
		self.use_residual = stride == 1 and in_channels == out_channels
		hidden_dim = in_channels * expand_ratio
		self.expand = (
			ConvBNAct(in_channels, hidden_dim, kernel_size=1)
			if expand_ratio != 1
			else nn.Identity()
		)
		self.dw_branch = ConvBNAct(
			hidden_dim,
			hidden_dim,
			kernel_size=kernel_size,
			stride=stride,
			groups=hidden_dim,
		)
		squeeze_channels = max(1, in_channels // 4)
		self.att_branch = nn.Sequential(
			SqueezeExcitation(hidden_dim, squeeze_channels),
			SimAM(lambda_value=simam_lambda),
		)
		self.project = ConvBNAct(hidden_dim * 2, out_channels, kernel_size=1, activation=False)
		self.block_dropout = nn.Dropout2d(float(block_dropout)) if block_dropout > 0 else nn.Identity()
		self.stochastic_depth = StochasticDepth(stochastic_depth_prob)

	def forward(self, x: Tensor) -> Tensor:
		identity = x
		x_expanded = self.expand(x)
		y_dw = self.dw_branch(x_expanded)
		y_att = self.att_branch(x_expanded)
		if y_att.shape[-2:] != y_dw.shape[-2:]:
			y_att = F.adaptive_avg_pool2d(y_att, output_size=y_dw.shape[-2:])
		y = torch.cat([y_dw, y_att], dim=1)
		y = self.project(y)
		y = self.block_dropout(y)
		if self.use_residual:
			y = self.stochastic_depth(y) + identity
		return y


# Это основная нейросетевая модель для классификации LPI-спектрограмм
class EfficientNetB0LPI(nn.Module):
	def __init__(
			self,
			num_classes: int = 13,
			in_channels: int = 1,
			improved: bool = True,
			dropout: float = 0.2,
			stochastic_depth_prob: float = 0.2,
			simam_lambda: float = 1e-4,
			block_dropout: float = 0.0,
	) -> None:
		super().__init__()
		block_cls = ImprovedMBConv if improved else BaselineMBConv
		layers = [ConvBNAct(in_channels, 32, kernel_size=3, stride=2)]
		total_blocks = sum(cfg.num_layers for cfg in B0_CONFIG)
		block_id = 0
		for cfg in B0_CONFIG:
			for i in range(cfg.num_layers):
				stride = cfg.stride if i == 0 else 1
				in_ch = cfg.input_channels if i == 0 else cfg.out_channels
				sd_prob = stochastic_depth_prob * block_id / max(1, total_blocks - 1)
				if improved:
					layers.append(
						block_cls(
							in_ch,
							cfg.out_channels,
							cfg.kernel,
							stride,
							cfg.expand_ratio,
							sd_prob,
							simam_lambda=simam_lambda,
							block_dropout=block_dropout
						)
					)
				else:
					layers.append(
						block_cls(
							in_ch,
							cfg.out_channels,
							cfg.kernel,
							stride,
							cfg.expand_ratio,
							sd_prob,
						)
					)
				block_id += 1
		layers.append(ConvBNAct(320, 1280, kernel_size=1))
		self.features = nn.Sequential(*layers)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Sequential(
			nn.Dropout(p=float(dropout)),
			nn.Linear(1280, int(num_classes)),
		)
		self._initialize_weights()

	# Инициализирует веса слоёв перед обучением
	def _initialize_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.kaiming_normal_(module.weight, mode="fan_out")
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.ones_(module.weight)
				nn.init.zeros_(module.bias)
			elif isinstance(module, nn.Linear):
				nn.init.normal_(module.weight, mean=0.0, std=0.01)
				nn.init.zeros_(module.bias)

	# Прямой проход: признаки EfficientNet -> глобальное усреднение -> линейная классификация
	def forward(self, x: Tensor) -> Tensor:
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		return self.classifier(x)


# Создаёт выбранный вариант модели по строковому имени архитектуры
def build_model(
		arch: str,
		num_classes: int = 13,
		in_channels: int = 1,
		dropout: float = 0.2,
		stochastic_depth_prob: float = 0.2,
		simam_lambda: float = 1e-4,
		block_dropout: float = 0.0,
) -> nn.Module:
	name = str(arch).lower().strip()
	if name in {"improved", "improved_b0", "improved_effnet_b0", "improved-efficientnet-b0"}:
		return EfficientNetB0LPI(
			num_classes=num_classes,
			in_channels=in_channels,
			improved=True,
			dropout=dropout,
			stochastic_depth_prob=stochastic_depth_prob,
			simam_lambda=simam_lambda,
			block_dropout=block_dropout,
		)
	if name in {"baseline", "baseline_b0", "efficientnet_b0", "effnet_b0"}:
		return EfficientNetB0LPI(
			num_classes=num_classes,
			in_channels=in_channels,
			improved=False,
			dropout=dropout,
			stochastic_depth_prob=stochastic_depth_prob,
			simam_lambda=simam_lambda,
			block_dropout=block_dropout,
		)
	raise ValueError("Unknown architecture: {}".format(arch))

# Возвращает общее и обучаемое число параметров модели
def count_parameters(model: nn.Module) -> Tuple[int, int]:
	total = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return int(total), int(trainable)
