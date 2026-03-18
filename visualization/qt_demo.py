#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChineseHarm 有害内容检测系统 — 桌面 GUI 应用
=============================================
基于 PyQt5，风格参考 YOLO 系列毕设演示界面。

功能:
  - 单条文本检测 / 批量文本文件导入
  - 检测结果可视化（类别标签、置信度进度条、多维标注）
  - 历史记录表格
  - Mock 模式（无需真实模型，答辩用）/ 远程 API 模式
  - 统计面板（本次会话的类别分布饼图）

运行方式:
    pip install PyQt5
    python qt_demo.py
"""

import sys
import os
import re
import json
import random
import time
import math
from pathlib import Path
from typing import Optional
from collections import Counter

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QPlainTextEdit, QLineEdit,
    QSplitter, QFrame, QGroupBox, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QStatusBar,
    QMessageBox, QComboBox, QStackedWidget, QTabWidget,
    QScrollArea, QDialog, QFormLayout, QDialogButtonBox,
    QAction, QMenuBar, QToolBar,
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QPropertyAnimation,
    QEasingCurve, QDateTime,
)
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QPixmap, QIcon, QPainter,
    QLinearGradient, QBrush, QPen, QTextCharFormat, QSyntaxHighlighter,
)

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────
APP_NAME    = "ChineseHarm 有害内容检测系统"
APP_VERSION = "v1.0 (毕业设计演示)"

CATEGORIES = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]
HARMFUL_CATS = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨"]

# 类别颜色（QSS 风格）
CAT_BG_COLORS = {
    "性别歧视":  "#FFEBEE",
    "种族歧视":  "#FFF3E0",
    "地域偏见":  "#FFFDE7",
    "LGBTQ歧视": "#F3E5F5",
    "其他仇恨":  "#E8EAF6",
    "无毒":      "#E8F5E9",
}
CAT_ACCENT_COLORS = {
    "性别歧视":  "#C62828",
    "种族歧视":  "#E65100",
    "地域偏见":  "#F9A825",
    "LGBTQ歧视": "#6A1B9A",
    "其他仇恨":  "#283593",
    "无毒":      "#1B5E20",
}
CAT_ICONS = {
    "性别歧视":  "⚧",
    "种族歧视":  "🌍",
    "地域偏见":  "📍",
    "LGBTQ歧视": "🏳️",
    "其他仇恨":  "⚠️",
    "无毒":      "✅",
}

# ─────────────────────────────────────────────────────────────────────────────
# Mock 检测引擎（答辩用，无需真实模型）
# ─────────────────────────────────────────────────────────────────────────────
class MockDetector:
    """基于规则的 Mock 检测器，模拟真实 Reviewer 模型的输出格式"""

    KEYWORD_MAP = {
        "性别歧视":  ["女的", "男的", "娘炮", "汉子", "妇女", "脑子不好使", "女人", "男人就是",
                     "女生不适合", "嫁得好", "女司机", "老公", "老婆"],
        "种族歧视":  ["黑人", "白人", "老外", "外国人", "黄种人", "种族", "鬼佬"],
        "地域偏见":  ["地区", "某地", "河南人", "东北人", "上海人", "北京人", "地域", "老家"],
        "LGBTQ歧视": ["lgbt", "同性恋", "变性", "弯的", "搞基", "gay", "les", "多元"],
        "其他仇恨":  ["垃圾", "废物", "滚", "死", "傻", "蠢", "贱", "烂", "恶心"],
    }
    IMPLICIT_KWS = ["所谓", "特色", "大开眼界", "真是", "以前多好", "无语", "笑死", "懂得都懂",
                    "你品你细品", "这种人", "这些人", "不是我说", "阴阳", "呵呵", "哦"]
    REPORTING_KWS = ["报道", "称", "曾", "据悉", "事件", "相关", "研究表明", "新闻", "消息"]

    def detect(self, text: str) -> dict:
        """返回结构化检测结果"""
        time.sleep(random.uniform(0.3, 0.8))  # 模拟推理延迟

        text_lower = text.lower()
        detected_cat  = "无毒"
        detected_expr = "非仇恨"
        toxic_type    = "无毒"
        confidence    = random.uniform(0.84, 0.97)

        for cat, kws in self.KEYWORD_MAP.items():
            if any(kw in text for kw in kws):
                detected_cat = cat
                if any(kw in text for kw in self.REPORTING_KWS):
                    detected_expr = "转述"
                    toxic_type    = "仇恨言论"
                    confidence    = random.uniform(0.65, 0.78)
                elif any(kw in text for kw in self.IMPLICIT_KWS):
                    detected_expr = "隐式仇恨"
                    toxic_type    = "仇恨言论"
                    confidence    = random.uniform(0.72, 0.86)
                else:
                    detected_expr = "显式仇恨"
                    toxic_type    = "一般攻击性" if cat == "其他仇恨" else "仇恨言论"
                    confidence    = random.uniform(0.88, 0.97)
                break

        is_harmful = detected_cat != "无毒"

        if is_harmful:
            raw_output = (
                f"判断：有害\n"
                f"类别：{detected_cat}\n"
                f"毒性类型：{toxic_type}\n"
                f"表达方式：{detected_expr}"
            )
        else:
            raw_output = "判断：无害"
            confidence = random.uniform(0.89, 0.97)

        return {
            "is_harmful":  is_harmful,
            "category":    detected_cat,
            "expression":  detected_expr,
            "toxic_type":  toxic_type,
            "confidence":  confidence,
            "raw_output":  raw_output,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 异步检测工作线程
# ─────────────────────────────────────────────────────────────────────────────
class DetectWorker(QThread):
    result_ready  = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress      = pyqtSignal(int, int)   # current, total

    def __init__(self, texts: list, mode: str, api_url: str = "", api_key: str = ""):
        super().__init__()
        self.texts   = texts
        self.mode    = mode
        self.api_url = api_url
        self.api_key = api_key
        self._detector = MockDetector()

    def run(self):
        for i, text in enumerate(self.texts):
            try:
                if self.mode == "mock":
                    result = self._detector.detect(text)
                else:
                    import requests
                    headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                    resp = requests.post(
                        self.api_url,
                        json={"text": text},
                        headers=headers,
                        timeout=30,
                    )
                    resp.raise_for_status()
                    rd = resp.json()
                    raw = rd.get("output", "")
                    result = {
                        "is_harmful": "有害" in raw,
                        "category":   rd.get("category", "未知"),
                        "expression": rd.get("expression", "未知"),
                        "toxic_type": rd.get("toxic_type", "未知"),
                        "confidence": rd.get("confidence", 0.5),
                        "raw_output": raw,
                    }
                result["text"]  = text
                result["index"] = i
                self.result_ready.emit(result)
                self.progress.emit(i + 1, len(self.texts))
            except Exception as e:
                self.error_occurred.emit(f"文本 {i+1} 检测失败: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 置信度进度条组件
# ─────────────────────────────────────────────────────────────────────────────
class ConfidenceBar(QWidget):
    def __init__(self, label: str, value: float, color: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        lbl = QLabel(label)
        lbl.setFixedWidth(90)
        lbl.setFont(QFont("Microsoft YaHei", 9))

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(value * 100))
        bar.setFixedHeight(18)
        bar.setTextVisible(False)
        bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 9px;
                background: #ECEFF1;
            }}
            QProgressBar::chunk {{
                border-radius: 9px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {color}99, stop:1 {color});
            }}
        """)

        val_lbl = QLabel(f"{value*100:.1f}%")
        val_lbl.setFixedWidth(48)
        val_lbl.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        val_lbl.setStyleSheet(f"color: {color};")

        layout.addWidget(lbl)
        layout.addWidget(bar, 1)
        layout.addWidget(val_lbl)


# ─────────────────────────────────────────────────────────────────────────────
# 结果卡片组件
# ─────────────────────────────────────────────────────────────────────────────
class ResultCard(QFrame):
    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        cat    = result["category"]
        expr   = result["expression"]
        ttype  = result["toxic_type"]
        conf   = result["confidence"]
        is_h   = result["is_harmful"]
        bg     = CAT_BG_COLORS.get(cat, "#FAFAFA")
        accent = CAT_ACCENT_COLORS.get(cat, "#555")
        icon   = CAT_ICONS.get(cat, "•")

        self.setStyleSheet(f"""
            ResultCard {{
                background: {bg};
                border: 2px solid {accent}44;
                border-left: 5px solid {accent};
                border-radius: 10px;
                padding: 4px;
            }}
        """)
        self.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(12, 10, 12, 10)

        # ── 判断结论 ──────────────────────────────────────────────────────────
        verdict = "⚠️ 有害内容" if is_h else "✅ 无害内容"
        verdict_lbl = QLabel(f"{icon}  {verdict}")
        verdict_lbl.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        verdict_lbl.setStyleSheet(f"color: {accent};")
        layout.addWidget(verdict_lbl)

        if is_h:
            # ── 类别标签 ──────────────────────────────────────────────────────
            tag_row = QHBoxLayout()
            for tag_text, tag_val in [("类别", cat), ("毒性类型", ttype), ("表达方式", expr)]:
                tag = QLabel(f"  {tag_text}：{tag_val}  ")
                tag.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
                tag.setStyleSheet(f"""
                    background: {accent};
                    color: white;
                    border-radius: 10px;
                    padding: 3px 6px;
                """)
                tag_row.addWidget(tag)
            tag_row.addStretch()
            layout.addLayout(tag_row)

        # ── 置信度进度条 ─────────────────────────────────────────────────────
        layout.addWidget(ConfidenceBar("置信度", conf, accent))

        # ── 表达方式提示 ──────────────────────────────────────────────────────
        if is_h:
            tip = ""
            if expr == "隐式仇恨":
                tip = "⚠️ 隐式仇恨：文本通过反讽、暗语、阴阳怪气等隐蔽方式传达歧视意图，是检测难点。"
            elif expr == "转述":
                tip = "ℹ️ 转述类：以新闻/讨论形式引用仇恨内容，立场模糊，需结合语境判断。"
            if tip:
                tip_lbl = QLabel(tip)
                tip_lbl.setWordWrap(True)
                tip_lbl.setFont(QFont("Microsoft YaHei", 8))
                tip_lbl.setStyleSheet(f"color: {accent}; background: transparent;")
                layout.addWidget(tip_lbl)


# ─────────────────────────────────────────────────────────────────────────────
# 设置对话框
# ─────────────────────────────────────────────────────────────────────────────
class SettingsDialog(QDialog):
    def __init__(self, parent=None, current_mode="mock", api_url="", api_key=""):
        super().__init__(parent)
        self.setWindowTitle("系统设置")
        self.setFixedSize(460, 260)
        self.setStyleSheet("background: white;")

        layout = QVBoxLayout(self)

        form = QFormLayout()
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignRight)

        self.mode_cb = QComboBox()
        self.mode_cb.addItems(["Mock 演示模式（无需模型）", "远程 API 模式"])
        self.mode_cb.setCurrentIndex(0 if current_mode == "mock" else 1)
        form.addRow("运行模式：", self.mode_cb)

        self.url_input = QLineEdit(api_url)
        self.url_input.setPlaceholderText("http://your-npu-server:8080/classify")
        form.addRow("API 地址：", self.url_input)

        self.key_input = QLineEdit(api_key)
        self.key_input.setEchoMode(QLineEdit.Password)
        self.key_input.setPlaceholderText("（可选）Bearer Token")
        form.addRow("API Key：", self.key_input)

        layout.addLayout(form)
        layout.addSpacing(16)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self):
        mode = "mock" if self.mode_cb.currentIndex() == 0 else "api"
        return mode, self.url_input.text(), self.key_input.text()


# ─────────────────────────────────────────────────────────────────────────────
# 主窗口
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}  {APP_VERSION}")
        self.setMinimumSize(1100, 720)
        self.resize(1280, 800)

        self.mode    = "mock"
        self.api_url = ""
        self.api_key = ""
        self._results_history = []
        self._worker: Optional[DetectWorker] = None

        self._setup_fonts()
        self._setup_style()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_central()
        self._setup_statusbar()

    # ── 字体 & 样式 ────────────────────────────────────────────────────────────
    def _setup_fonts(self):
        self.font_main  = QFont("Microsoft YaHei", 10)
        self.font_title = QFont("Microsoft YaHei", 11, QFont.Bold)
        self.font_mono  = QFont("Consolas", 9)
        QApplication.setFont(self.font_main)

    def _setup_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #F5F6FA; }
            QGroupBox {
                font-weight: bold; font-size: 10pt;
                border: 1.5px solid #DDE1E7;
                border-radius: 8px; margin-top: 8px;
                background: white; padding: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 0 4px;
                color: #1565C0;
            }
            QPushButton {
                border-radius: 6px; padding: 7px 18px;
                font-size: 10pt; font-family: "Microsoft YaHei";
                border: none;
            }
            QPushButton#btnDetect {
                background: #1565C0; color: white; font-weight: bold;
                font-size: 11pt;
            }
            QPushButton#btnDetect:hover  { background: #1976D2; }
            QPushButton#btnDetect:pressed { background: #0D47A1; }
            QPushButton#btnDetect:disabled { background: #B0BEC5; }
            QPushButton#btnLoad {
                background: #E8EAF6; color: #283593;
            }
            QPushButton#btnLoad:hover { background: #C5CAE9; }
            QPushButton#btnClear {
                background: #FFEBEE; color: #C62828;
            }
            QPushButton#btnClear:hover { background: #FFCDD2; }
            QTextEdit, QPlainTextEdit {
                border: 1.5px solid #DDE1E7; border-radius: 6px;
                padding: 6px; background: white; font-size: 10pt;
            }
            QTextEdit:focus, QPlainTextEdit:focus {
                border-color: #1565C0;
            }
            QTableWidget {
                border: 1.5px solid #DDE1E7; border-radius: 6px;
                background: white; gridline-color: #F0F0F0;
            }
            QTableWidget::item { padding: 4px 8px; }
            QTableWidget::item:selected { background: #E3F2FD; color: #0D47A1; }
            QHeaderView::section {
                background: #F5F6FA; font-weight: bold; color: #333;
                border: none; border-bottom: 1px solid #DDE1E7;
                padding: 6px 8px;
            }
            QTabWidget::pane { border: 1.5px solid #DDE1E7; border-radius: 0 8px 8px 8px; background: white; }
            QTabBar::tab {
                padding: 7px 18px; font-size: 10pt;
                border: 1.5px solid transparent; border-bottom: none;
                border-radius: 6px 6px 0 0; margin-right: 2px;
            }
            QTabBar::tab:selected { background: white; border-color: #DDE1E7; color: #1565C0; font-weight: bold; }
            QTabBar::tab:!selected { background: #ECEFF1; color: #607D8B; }
            QScrollBar:vertical {
                width: 8px; background: #F5F6FA; border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #CFD8DC; border-radius: 4px; min-height: 30px;
            }
            QStatusBar { background: #1565C0; color: white; font-size: 9pt; }
            QLabel#mode_badge {
                background: #E8F5E9; color: #1B5E20;
                border-radius: 8px; padding: 2px 10px;
                font-weight: bold; font-size: 9pt;
            }
            QToolBar {
                background: white; border-bottom: 1px solid #DDE1E7;
                spacing: 6px; padding: 4px;
            }
        """)

    # ── 菜单栏 ─────────────────────────────────────────────────────────────────
    def _setup_menu(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar { background: white; font-size: 10pt; }
            QMenuBar::item:selected { background: #E3F2FD; }
            QMenu { background: white; border: 1px solid #DDE1E7; }
            QMenu::item:selected { background: #E3F2FD; color: #0D47A1; }
        """)

        file_menu = menubar.addMenu("文件(&F)")
        file_menu.addAction("📂 加载文本文件", self._load_file, "Ctrl+O")
        file_menu.addAction("💾 导出检测记录", self._export_history, "Ctrl+S")
        file_menu.addSeparator()
        file_menu.addAction("退出(&Q)", self.close, "Ctrl+Q")

        settings_menu = menubar.addMenu("设置(&S)")
        settings_menu.addAction("⚙️ 系统配置", self._open_settings)

        help_menu = menubar.addMenu("帮助(&H)")
        help_menu.addAction("📖 关于", self._show_about)

    # ── 工具栏 ─────────────────────────────────────────────────────────────────
    def _setup_toolbar(self):
        tb = QToolBar("主工具栏", self)
        tb.setIconSize(QSize(20, 20))
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addAction("📂 加载文件", self._load_file)
        tb.addSeparator()
        tb.addAction("💾 导出记录", self._export_history)
        tb.addSeparator()
        tb.addAction("🗑️ 清空记录", self._clear_all)
        tb.addSeparator()
        tb.addAction("⚙️ 设置", self._open_settings)

        # 右侧模式标签
        from PyQt5.QtWidgets import QSizePolicy
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self._mode_badge = QLabel("🎭 Mock 演示模式")
        self._mode_badge.setObjectName("mode_badge")
        tb.addWidget(self._mode_badge)
        tb.addWidget(QLabel("  "))

    # ── 中央区域 ───────────────────────────────────────────────────────────────
    def _setup_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(10)

        # ── 左侧：输入区 ───────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(420)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # 标题 banner
        banner = QLabel("🛡️  中文有害内容检测系统")
        banner.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        banner.setStyleSheet("""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #1565C0, stop:1 #283593);
            color: white; border-radius: 8px;
            padding: 10px 14px;
        """)
        left_layout.addWidget(banner)

        # 输入框组
        input_group = QGroupBox("📝 输入文本")
        ig_layout = QVBoxLayout(input_group)
        ig_layout.setSpacing(6)

        self._input_edit = QTextEdit()
        self._input_edit.setPlaceholderText(
            "请输入待检测的中文文本...\n\n"
            "例如：\n"
            "女的就是脑子不好使，别来搞技术了。\n\n"
            "支持多行输入（每行作为一条独立文本检测）"
        )
        self._input_edit.setFont(self.font_main)
        self._input_edit.setMinimumHeight(180)
        ig_layout.addWidget(self._input_edit)

        # 预设样例下拉
        example_row = QHBoxLayout()
        example_lbl = QLabel("快速样例：")
        self._example_cb = QComboBox()
        self._example_cb.addItems([
            "—— 选择预设样例 ——",
            "显式性别歧视",
            "隐式地域偏见",
            "转述类文本",
            "LGBTQ歧视（隐式）",
            "无毒文本",
        ])
        self._example_cb.currentIndexChanged.connect(self._fill_example)
        example_row.addWidget(example_lbl)
        example_row.addWidget(self._example_cb, 1)
        ig_layout.addLayout(example_row)
        left_layout.addWidget(input_group)

        # 操作按钮
        btn_row = QHBoxLayout()
        self._btn_load = QPushButton("📂 加载文件")
        self._btn_load.setObjectName("btnLoad")
        self._btn_load.clicked.connect(self._load_file)

        self._btn_clear_input = QPushButton("🗑️ 清空")
        self._btn_clear_input.setObjectName("btnClear")
        self._btn_clear_input.clicked.connect(self._input_edit.clear)

        btn_row.addWidget(self._btn_load)
        btn_row.addWidget(self._btn_clear_input)
        left_layout.addLayout(btn_row)

        self._btn_detect = QPushButton("🔍  开始检测")
        self._btn_detect.setObjectName("btnDetect")
        self._btn_detect.setFixedHeight(46)
        self._btn_detect.clicked.connect(self._start_detect)
        left_layout.addWidget(self._btn_detect)

        # 进度条
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(10)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        self._progress.setStyleSheet("""
            QProgressBar { border: none; border-radius: 5px; background: #ECEFF1; }
            QProgressBar::chunk { border-radius: 5px; background: #1565C0; }
        """)
        left_layout.addWidget(self._progress)

        # 模型说明卡片
        info_box = QGroupBox("ℹ️ 系统说明")
        info_layout = QVBoxLayout(info_box)
        info_text = QLabel(
            "本系统基于 Stackelberg 对抗自对弈训练框架，\n"
            "Reviewer 模型（Qwen-3B/7B + LoRA）对文本进行\n"
            "多维度分类检测，覆盖 6 类中文有害内容：\n\n"
            "  ⚧ 性别歧视  🌍 种族歧视  📍 地域偏见\n"
            "  🏳️ LGBTQ歧视  ⚠️ 其他仇恨  ✅ 无毒\n\n"
            "支持显式/隐式/转述三种表达方式识别。"
        )
        info_text.setFont(QFont("Microsoft YaHei", 9))
        info_text.setStyleSheet("color: #555; background: transparent; line-height: 1.6;")
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        left_layout.addWidget(info_box)
        left_layout.addStretch()

        main_layout.addWidget(left)

        # ── 右侧：结果区（Tab）──────────────────────────────────────────────────
        self._tabs = QTabWidget()
        main_layout.addWidget(self._tabs, 1)

        # Tab 1: 最新结果
        self._result_scroll = QScrollArea()
        self._result_scroll.setWidgetResizable(True)
        self._result_scroll.setFrameShape(QFrame.NoFrame)
        self._result_content = QWidget()
        self._result_vbox = QVBoxLayout(self._result_content)
        self._result_vbox.setAlignment(Qt.AlignTop)
        self._result_vbox.setContentsMargins(12, 12, 12, 12)
        self._result_vbox.setSpacing(10)
        self._result_scroll.setWidget(self._result_content)
        self._tabs.addTab(self._result_scroll, "🔍 检测结果")

        # Tab 2: 原始输出
        self._raw_output = QPlainTextEdit()
        self._raw_output.setReadOnly(True)
        self._raw_output.setFont(self.font_mono)
        self._raw_output.setStyleSheet("background: #FAFAFA; color: #333;")
        self._tabs.addTab(self._raw_output, "📄 原始输出")

        # Tab 3: 历史记录表
        self._history_table = QTableWidget()
        self._history_table.setColumnCount(6)
        self._history_table.setHorizontalHeaderLabels(
            ["时间", "文本（截取）", "判断", "类别", "表达方式", "置信度"])
        self._history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._history_table.setAlternatingRowColors(True)
        self._history_table.setStyleSheet("""
            QTableWidget { alternate-background-color: #F8F9FA; }
        """)
        self._tabs.addTab(self._history_table, "📋 历史记录")

        # Tab 4: 统计面板（自绘）
        self._stats_widget = StatsPanel()
        self._tabs.addTab(self._stats_widget, "📊 统计面板")

        # 默认占位
        self._show_placeholder()

    # ── 状态栏 ─────────────────────────────────────────────────────────────────
    def _setup_statusbar(self):
        sb = self.statusBar()
        self._status_label = QLabel("就绪  |  等待输入文本…")
        self._status_label.setStyleSheet("color: white;")
        sb.addPermanentWidget(self._status_label, 1)

        self._total_label = QLabel("总检测：0 条  |  有害：0 条  |  无害：0 条")
        self._total_label.setStyleSheet("color: #B3E5FC; padding-right: 10px;")
        sb.addPermanentWidget(self._total_label)

    # ── 槽函数 —————————————————————————————————————————————————————————————────
    def _fill_example(self, idx):
        examples = {
            1: "女的就是脑子不好使，别来搞技术了。",
            2: "某地区的司机真的很有"特色"，每次都让人大开眼界，懂得都懂。",
            3: "有报道称，该社区曾多次发生与地域身份相关的言语冲突事件。",
            4: "这种所谓的多元包容真是让人无语，以前的社会多好啊。",
            5: "今天天气真好，适合出去散步，好好放松一下。",
        }
        if idx in examples:
            self._input_edit.setPlainText(examples[idx])

    def _show_placeholder(self):
        ph = QLabel("检测结果将在此显示\n\n请在左侧输入文本后点击「开始检测」")
        ph.setFont(QFont("Microsoft YaHei", 12))
        ph.setAlignment(Qt.AlignCenter)
        ph.setStyleSheet("color: #B0BEC5;")
        ph.setObjectName("placeholder")
        self._result_vbox.addWidget(ph)

    def _start_detect(self):
        raw = self._input_edit.toPlainText().strip()
        if not raw:
            QMessageBox.warning(self, "提示", "请先输入待检测的文本。")
            return

        texts = [t.strip() for t in raw.splitlines() if t.strip()]
        if not texts:
            return

        # 清除旧结果
        while self._result_vbox.count():
            item = self._result_vbox.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._raw_output.clear()

        self._btn_detect.setEnabled(False)
        self._btn_detect.setText("⏳  检测中…")
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._set_status(f"正在检测 {len(texts)} 条文本…")

        self._worker = DetectWorker(texts, self.mode, self.api_url, self.api_key)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_result(self, result: dict):
        # ── 结果卡片 ──────────────────────────────────────────────────────────
        # 文本标签
        text_lbl = QLabel(f"【文本 {result['index']+1}】{result['text'][:80]}"
                          + ("…" if len(result['text']) > 80 else ""))
        text_lbl.setFont(QFont("Microsoft YaHei", 9))
        text_lbl.setStyleSheet("color: #555; padding: 0 4px;")
        text_lbl.setWordWrap(True)
        self._result_vbox.addWidget(text_lbl)

        card = ResultCard(result)
        self._result_vbox.addWidget(card)

        # ── 原始输出 ──────────────────────────────────────────────────────────
        sep = "─" * 50
        self._raw_output.appendPlainText(
            f"\n[文本 {result['index']+1}] {result['text'][:60]}…\n"
            f"{sep}\n{result['raw_output']}\n"
        )

        # ── 历史表格 ──────────────────────────────────────────────────────────
        row = self._history_table.rowCount()
        self._history_table.insertRow(row)
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        items_data = [
            timestamp,
            result['text'][:40] + ("…" if len(result['text']) > 40 else ""),
            "⚠️ 有害" if result['is_harmful'] else "✅ 无害",
            result['category'],
            result['expression'],
            f"{result['confidence']*100:.1f}%",
        ]
        for col, data in enumerate(items_data):
            item = QTableWidgetItem(data)
            item.setTextAlignment(Qt.AlignCenter)
            if col == 2:
                item.setForeground(QColor(CAT_ACCENT_COLORS.get(result['category'], "#333")))
                item.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
            if col == 3:
                item.setBackground(QColor(CAT_BG_COLORS.get(result['category'], "#FAFAFA")))
            self._history_table.setItem(row, col, item)

        # ── 统计更新 ──────────────────────────────────────────────────────────
        self._results_history.append(result)
        self._stats_widget.update_stats(self._results_history)
        self._update_totals()

        # 滚动到底部
        self._result_scroll.verticalScrollBar().setValue(
            self._result_scroll.verticalScrollBar().maximum()
        )
        self._tabs.setCurrentIndex(0)

    def _on_error(self, msg: str):
        err_lbl = QLabel(f"❌ {msg}")
        err_lbl.setStyleSheet("color: #C62828; padding: 4px 8px;")
        self._result_vbox.addWidget(err_lbl)

    def _on_progress(self, current: int, total: int):
        self._progress.setValue(int(current / total * 100))
        self._set_status(f"正在检测… {current}/{total}")

    def _on_finished(self):
        self._btn_detect.setEnabled(True)
        self._btn_detect.setText("🔍  开始检测")
        self._progress.setVisible(False)
        total   = len(self._results_history)
        harmful = sum(1 for r in self._results_history if r['is_harmful'])
        self._set_status(f"✅ 检测完成，共 {total} 条历史记录")
        self._update_totals()

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "加载文本文件", "", "文本文件 (*.txt *.csv *.jsonl *.json);;所有文件 (*)"
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                lines = [L.strip() for L in f if L.strip()]
            # JSONL 格式兼容
            texts = []
            for line in lines[:100]:  # 限制100条
                try:
                    obj = json.loads(line)
                    texts.append(obj.get("文本") or obj.get("text") or obj.get("content") or line)
                except json.JSONDecodeError:
                    texts.append(line)
            self._input_edit.setPlainText("\n".join(texts))
            self._set_status(f"已加载文件：{Path(path).name}，共 {len(texts)} 条（最多取前100条）")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def _export_history(self):
        if not self._results_history:
            QMessageBox.information(self, "提示", "暂无检测记录可导出。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出检测记录", "detection_results.jsonl", "JSONL (*.jsonl);;JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                for r in self._results_history:
                    f.write(json.dumps({
                        "text":       r["text"],
                        "is_harmful": r["is_harmful"],
                        "category":   r["category"],
                        "expression": r["expression"],
                        "toxic_type": r["toxic_type"],
                        "confidence": round(r["confidence"], 4),
                    }, ensure_ascii=False) + "\n")
            self._set_status(f"✅ 已导出 {len(self._results_history)} 条记录到 {Path(path).name}")
            QMessageBox.information(self, "导出成功", f"已导出到：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _clear_all(self):
        reply = QMessageBox.question(self, "确认", "确定清空所有检测记录？", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._results_history.clear()
            while self._result_vbox.count():
                item = self._result_vbox.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self._raw_output.clear()
            self._history_table.setRowCount(0)
            self._stats_widget.update_stats([])
            self._update_totals()
            self._show_placeholder()
            self._set_status("已清空所有记录")

    def _open_settings(self):
        dlg = SettingsDialog(self, self.mode, self.api_url, self.api_key)
        if dlg.exec_() == QDialog.Accepted:
            self.mode, self.api_url, self.api_key = dlg.get_values()
            if self.mode == "mock":
                self._mode_badge.setText("🎭 Mock 演示模式")
                self._mode_badge.setStyleSheet(
                    "background: #E8F5E9; color: #1B5E20; border-radius: 8px;"
                    " padding: 2px 10px; font-weight: bold; font-size: 9pt;")
            else:
                self._mode_badge.setText("🌐 远程 API 模式")
                self._mode_badge.setStyleSheet(
                    "background: #E3F2FD; color: #0D47A1; border-radius: 8px;"
                    " padding: 2px 10px; font-weight: bold; font-size: 9pt;")
            self._set_status(f"已切换到 {'Mock 演示' if self.mode=='mock' else '远程 API'} 模式")

    def _show_about(self):
        QMessageBox.about(self, "关于",
            f"<h3>{APP_NAME}</h3>"
            f"<p>{APP_VERSION}</p>"
            "<p>本系统为毕业设计演示程序，实现了基于<br>"
            "<b>Stackelberg 对抗自对弈强化学习</b>的<br>"
            "中文有害内容多维度检测。</p>"
            "<hr>"
            "<p>覆盖类别：性别歧视 / 种族歧视 / 地域偏见<br>"
            "LGBTQ歧视 / 其他仇恨 / 无毒</p>"
            "<p>数据集：ToxiCN (ACL 2023)</p>"
            "<p>训练框架：TRL + DeepSpeed (昇腾 910B)</p>"
        )

    def _set_status(self, text: str):
        self._status_label.setText(f"  {text}")

    def _update_totals(self):
        total   = len(self._results_history)
        harmful = sum(1 for r in self._results_history if r['is_harmful'])
        safe    = total - harmful
        self._total_label.setText(
            f"总检测：{total} 条  |  有害：{harmful} 条  |  无害：{safe} 条"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 统计面板（自绘饼图 + 条形图）
# ─────────────────────────────────────────────────────────────────────────────
class StatsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._results = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        title = QLabel("📊 本次会话统计")
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title.setStyleSheet("color: #1565C0;")
        layout.addWidget(title)

        # 数字概览
        self._overview = QLabel("暂无数据，请先进行检测。")
        self._overview.setFont(QFont("Microsoft YaHei", 10))
        self._overview.setStyleSheet("color: #555;")
        layout.addWidget(self._overview)

        # 类别分布（自绘进度条）
        dist_group = QGroupBox("分类分布")
        self._dist_layout = QVBoxLayout(dist_group)
        layout.addWidget(dist_group)

        # 表达方式分布
        expr_group = QGroupBox("表达方式分布")
        self._expr_layout = QVBoxLayout(expr_group)
        layout.addWidget(expr_group)

        layout.addStretch()

    def update_stats(self, results: list):
        self._results = results

        # 清除旧控件
        for layout in [self._dist_layout, self._expr_layout]:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        if not results:
            self._overview.setText("暂无数据，请先进行检测。")
            return

        total   = len(results)
        harmful = sum(1 for r in results if r['is_harmful'])
        safe    = total - harmful

        self._overview.setText(
            f"检测总量：{total} 条  |  有害：{harmful} 条（{harmful/total*100:.1f}%）"
            f"  |  无害：{safe} 条（{safe/total*100:.1f}%）"
        )

        # 类别分布
        cat_counter = Counter(r['category'] for r in results)
        for cat in ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]:
            cnt = cat_counter.get(cat, 0)
            if total > 0:
                ratio = cnt / total
                bar = ConfidenceBar(
                    f"{CAT_ICONS.get(cat,'')} {cat}",
                    ratio,
                    CAT_ACCENT_COLORS.get(cat, "#888"),
                )
                # 在进度条右侧追加计数
                count_lbl = QLabel(f"{cnt} 条")
                count_lbl.setFont(QFont("Microsoft YaHei", 8))
                count_lbl.setStyleSheet("color: #888;")
                row = QHBoxLayout()
                row.addWidget(bar, 1)
                row.addWidget(count_lbl)
                wrapper = QWidget()
                wrapper.setLayout(row)
                self._dist_layout.addWidget(wrapper)

        # 表达方式分布
        expr_counter = Counter(r['expression'] for r in results)
        expr_info = {
            "显式仇恨": ("#EF553B", "直接含有歧视性关键词"),
            "隐式仇恨": ("#FF7F0E", "反讽/暗语等隐蔽方式 ← 检测难点"),
            "转述":     ("#636EFA", "引用仇恨内容但立场模糊"),
            "非仇恨":   ("#00CC96", "无害表达"),
        }
        for expr, (color, desc) in expr_info.items():
            cnt = expr_counter.get(expr, 0)
            ratio = cnt / total if total > 0 else 0
            bar = ConfidenceBar(expr, ratio, color)
            cnt_lbl = QLabel(f"{cnt}  {desc}")
            cnt_lbl.setFont(QFont("Microsoft YaHei", 8))
            cnt_lbl.setStyleSheet("color: #888;")
            self._expr_layout.addWidget(bar)
            self._expr_layout.addWidget(cnt_lbl)


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")

    # 全局 Fusion 调色板（保证白色背景）
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#F5F6FA"))
    palette.setColor(QPalette.WindowText, QColor("#212121"))
    palette.setColor(QPalette.Base, QColor("#FFFFFF"))
    palette.setColor(QPalette.AlternateBase, QColor("#F8F9FA"))
    palette.setColor(QPalette.Highlight, QColor("#1565C0"))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
