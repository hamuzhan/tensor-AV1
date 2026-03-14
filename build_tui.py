#!/usr/bin/env python3
"""
TENSOR-AV1 Build System TUI
Interactive terminal interface for configuring and building SVT-AV1.
"""

import curses
import subprocess
import os
import re
import time
import shutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(PROJECT_DIR, "Build")
BIN_DIR = os.path.join(PROJECT_DIR, "Bin")

# ── Compact logo for narrow terminals (<80 cols) ──
LOGO_SMALL = [
    " TENSOR-AV1 ",
    " Build System ",
]

# ── Full logo (>=80 cols) ──
LOGO_FULL = [
    "  _____                              _  __   ___",
    " |_   _|__ _ __  ___  ___  _ __     / \\|  \\ / _ \\",
    "   | |/ _ \\ '_ \\/ __|/ _ \\| '__|   / _ \\ \\ / / |",
    "   | |  __/ | | \\__ \\ (_) | |  _  / ___ \\ V /| |",
    "   |_|\\___|_| |_|___/\\___/|_| (_)/_/   \\_\\_/  \\___\\",
]

# ── Wide logo (>=110 cols) ──
LOGO_WIDE = [
    "  _______ ______ _   _  _____  ____  _____            __      __ __ ",
    " |__   __|  ____| \\ | |/ ____|/ __ \\|  __ \\          /\\ \\    / //_ |",
    "    | |  | |__  |  \\| | (___ | |  | | |__) | ______ /  \\ \\  / /  | |",
    "    | |  |  __| | . ` |\\___ \\| |  | |  _  / |______/ /\\ \\ \\/ /   | |",
    "    | |  | |____| |\\  |____) | |__| | | \\ \\       / ____ \\  /    | |",
    "    |_|  |______|_| \\_|_____/ \\____/|_|  \\_\\     /_/    \\_\\/     |_|",
]

SUBTITLE = "Scalable Video Technology for AV1  //  ML Data Collection Pipeline"

# ── Box Drawing ──
BOX_H  = "\u2500"   # ─
BOX_V  = "\u2502"   # │
BOX_TL = "\u256d"   # ╭
BOX_TR = "\u256e"   # ╮
BOX_BL = "\u2570"   # ╰
BOX_BR = "\u256f"   # ╯
BOX_LT = "\u251c"   # ├
BOX_RT = "\u2524"   # ┤
BULLET = "\u25b8"   # ▸
CHECK  = "\u25c9"   # ◉
EMPTY  = "\u25cb"   # ○
ARROW  = "\u25b6"   # ▶


class Option:
    def __init__(self, key, label, cmake_flag, default=False, group="feature"):
        self.key = key
        self.label = label
        self.cmake_flag = cmake_flag
        self.enabled = default
        self.group = group

OPTIONS = [
    Option("data_collection", "Data Collection (HDF5 ML Pipeline)", "ENABLE_DATA_COLLECTION", True, "feature"),
    Option("native",          "Native CPU Tuning (-march=native)",  "NATIVE",                 True, "optimize"),
    Option("lto",             "Link-Time Optimization (LTO)",       "SVT_AV1_LTO",            True, "optimize"),
    Option("shared",          "Shared Libraries (.so)",             "BUILD_SHARED_LIBS",       True, "build"),
    Option("apps",            "Encoder Application",                "BUILD_APPS",              True, "build"),
    Option("testing",         "Tests (Unit / API / E2E)",           "BUILD_TESTING",          False, "build"),
    Option("minimal",         "Minimal Build",                      "MINIMAL_BUILD",          False, "feature"),
    Option("rtc",             "RTC Mode (Real-Time)",               "RTC_BUILD",              False, "feature"),
    Option("c_only",          "C-Only (No SIMD)",                   "COMPILE_C_ONLY",         False, "feature"),
    Option("quiet",           "Quiet Logging",                      "LOG_QUIET",              False, "feature"),
]

BUILD_TYPES = ["Release", "Debug", "RelWithDebInfo", "MinSizeRel"]

PRESETS = {
    "Development": {
        "desc": "Debug build with tests enabled, no heavy optimizations",
        "build_type": "Debug", "compiler": "gcc-14",
        "options": {"data_collection": True, "native": False, "lto": False,
                    "shared": True, "apps": True, "testing": True,
                    "minimal": False, "rtc": False, "c_only": False, "quiet": False},
    },
    "Release Optimized": {
        "desc": "Maximum performance, no data collection overhead",
        "build_type": "Release", "compiler": "gcc-14",
        "options": {"data_collection": False, "native": True, "lto": True,
                    "shared": True, "apps": True, "testing": False,
                    "minimal": False, "rtc": False, "c_only": False, "quiet": False},
    },
    "Data Collection": {
        "desc": "Full optimization + HDF5 ML data pipeline for Mamba-2 training",
        "build_type": "Release", "compiler": "gcc-14",
        "options": {"data_collection": True, "native": True, "lto": True,
                    "shared": True, "apps": True, "testing": False,
                    "minimal": False, "rtc": False, "c_only": False, "quiet": False},
    },
    "Minimal / Quick": {
        "desc": "Fastest compile time, C-only, minimal features",
        "build_type": "Release", "compiler": "gcc",
        "options": {"data_collection": False, "native": False, "lto": False,
                    "shared": True, "apps": True, "testing": False,
                    "minimal": True, "rtc": False, "c_only": True, "quiet": True},
    },
}


def detect_cores():
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4

def detect_compilers():
    compilers = []
    for cc, cxx, label in [
        ("gcc-14", "g++-14", "GCC 14"),
        ("gcc", "g++", "GCC (system)"),
        ("clang", "clang++", "Clang"),
    ]:
        if shutil.which(cc):
            try:
                ver = subprocess.check_output([cc, "--version"], stderr=subprocess.DEVNULL,
                                              text=True).split("\n")[0].strip()
                compilers.append((cc, cxx, f"{label}  {ver}"))
            except Exception:
                compilers.append((cc, cxx, label))
    return compilers


class TUI:
    def __init__(self, stdscr):
        self.scr = stdscr
        self.options = [Option(o.key, o.label, o.cmake_flag, o.enabled, o.group) for o in OPTIONS]
        self.compilers = detect_compilers()
        self.compiler_idx = 0
        self.bt_idx = 0
        self.cores = detect_cores()
        self.clean = True
        self._init_colors()

    def _init_colors(self):
        curses.start_color()
        curses.use_default_colors()
        curses.curs_set(0)
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, 8, -1)                        # dim gray
        curses.init_pair(4, curses.COLOR_YELLOW, -1)
        curses.init_pair(5, curses.COLOR_WHITE, 4)         # status bar (white on dark yellow)
        curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_GREEN)
        curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_RED)
        curses.init_pair(8, curses.COLOR_MAGENTA, -1)
        curses.init_pair(9, curses.COLOR_RED, -1)
        curses.init_pair(10, curses.COLOR_BLUE, -1)

    C_ACCENT = 1
    C_ON     = 2
    C_DIM    = 3
    C_HL     = 4
    C_BAR    = 5
    C_OK     = 6
    C_ERR    = 7
    C_LOGO   = 8
    C_RED    = 9
    C_BLUE   = 10

    # ── Drawing primitives ──

    def w(self):
        return self.scr.getmaxyx()[1]

    def h(self):
        return self.scr.getmaxyx()[0]

    def put(self, y, x, text, color=0, bold=False):
        h, w = self.scr.getmaxyx()
        if y < 0 or y >= h or x >= w:
            return
        attr = curses.color_pair(color)
        if bold:
            attr |= curses.A_BOLD
        try:
            self.scr.addnstr(y, x, text, w - x - 1, attr)
        except curses.error:
            pass

    def hline(self, y, label=""):
        w = self.w()
        if label:
            pad = max(0, (w - len(label) - 4) // 2)
            line = BOX_H * pad + " " + label + " " + BOX_H * max(0, w - pad - len(label) - 2)
        else:
            line = BOX_H * w
        self.put(y, 0, line[:w], self.C_DIM)

    def status_bar(self, text):
        h, w = self.scr.getmaxyx()
        bar = " " + text.ljust(w - 2)
        self.put(h - 1, 0, bar[:w - 1], self.C_BAR)

    def center(self, y, text, color=0, bold=False):
        x = max(0, (self.w() - len(text)) // 2)
        self.put(y, x, text, color, bold)

    def draw_logo(self, y=0):
        w = self.w()
        if w >= 110:
            logo = LOGO_WIDE
        elif w >= 80:
            logo = LOGO_FULL
        else:
            logo = LOGO_SMALL
        for i, line in enumerate(logo):
            self.center(y + i, line, self.C_LOGO, bold=True)
        y += len(logo)
        if w >= 60:
            self.center(y, SUBTITLE, self.C_DIM)
            y += 1
        return y + 1

    # ── Pages ──

    def run(self):
        page = "main"
        while True:
            if page == "main":
                page = self.page_main()
            elif page == "config":
                page = self.page_config()
            elif page == "presets":
                page = self.page_presets()
            elif page == "build":
                self.page_build()
                page = "main"
            elif page == "quit":
                break

    # ── Main Menu ──

    def page_main(self):
        items = [
            (f"{ARROW} Configure & Build",       "config"),
            (f"{ARROW} Quick Presets",            "presets"),
            (f"{ARROW} Build Current Settings",   "build"),
            (f"{ARROW} Quit",                     "quit"),
        ]
        sel = 0
        while True:
            self.scr.erase()
            y = self.draw_logo(1)
            y += 1
            self.hline(y, "Main Menu")
            y += 2
            for i, (label, _) in enumerate(items):
                if i == sel:
                    self.put(y, 6, f"  {label}  ", self.C_HL, bold=True)
                else:
                    self.put(y, 6, f"  {label}  ", self.C_DIM)
                y += 2
            self.status_bar(" Up/Down: navigate  |  Enter: select  |  q: quit")
            self.scr.refresh()
            k = self.scr.getch()
            if k == curses.KEY_UP:
                sel = (sel - 1) % len(items)
            elif k == curses.KEY_DOWN:
                sel = (sel + 1) % len(items)
            elif k in (10, 13, curses.KEY_ENTER):
                return items[sel][1]
            elif k == ord("q"):
                return "quit"

    # ── Configuration ──

    def page_config(self):
        sel = 0
        scroll = 0
        while True:
            self.scr.erase()
            h, w = self.scr.getmaxyx()
            y = 0
            self.center(y, " TENSOR-AV1 Build Configuration ", self.C_LOGO, bold=True)
            y += 2

            rows = []
            # Settings
            cc, cxx, clabel = self.compilers[self.compiler_idx]
            bt = BUILD_TYPES[self.bt_idx]
            rows.append(("setting", "compiler",   f"Compiler      {BOX_V}  {clabel}"))
            rows.append(("setting", "build_type", f"Build Type    {BOX_V}  {bt}"))
            rows.append(("setting", "cores",      f"Cores         {BOX_V}  {self.cores} / {detect_cores()}"))
            rows.append(("setting", "clean",      f"Clean Build   {BOX_V}  {'Yes' if self.clean else 'No'}"))
            rows.append(("sep", None, "Optimization"))
            for o in self.options:
                if o.group == "optimize":
                    rows.append(("option", o, None))
            rows.append(("sep", None, "Features"))
            for o in self.options:
                if o.group == "feature":
                    rows.append(("option", o, None))
            rows.append(("sep", None, "Build Targets"))
            for o in self.options:
                if o.group == "build":
                    rows.append(("option", o, None))
            rows.append(("sep", None, ""))
            rows.append(("action", "build", f"  {ARROW} START BUILD"))
            rows.append(("action", "back",  f"  {ARROW} Back to Main Menu"))

            selectable = [i for i, (t, _, _) in enumerate(rows) if t in ("setting", "option", "action")]
            if sel >= len(selectable):
                sel = len(selectable) - 1

            # Viewport
            viewport_top = y + 1
            viewport_h = h - viewport_top - 2  # reserve status bar + margin
            # Auto-scroll to keep selection visible
            sel_row = selectable[sel]
            if sel_row - scroll >= viewport_h:
                scroll = sel_row - viewport_h + 1
            if sel_row < scroll:
                scroll = sel_row

            for ri in range(scroll, min(len(rows), scroll + viewport_h)):
                ry = viewport_top + (ri - scroll)
                if ry >= h - 1:
                    break

                kind, data, text = rows[ri]
                is_sel = ri == selectable[sel]

                if kind == "sep":
                    self.hline(ry, text)
                elif kind == "setting":
                    marker = f" {BULLET} " if is_sel else "   "
                    c = self.C_HL if is_sel else 0
                    self.put(ry, 3, marker, self.C_HL if is_sel else self.C_DIM, bold=is_sel)
                    self.put(ry, 6, text, c, bold=is_sel)
                    if is_sel:
                        hint = " <Left/Right>"
                        self.put(ry, 6 + len(text) + 1, hint, self.C_DIM)
                elif kind == "option":
                    o = data
                    marker = f" {BULLET} " if is_sel else "   "
                    icon = CHECK if o.enabled else EMPTY
                    c = self.C_ON if o.enabled else self.C_DIM
                    self.put(ry, 3, marker, self.C_HL if is_sel else self.C_DIM, bold=is_sel)
                    self.put(ry, 6, f"{icon} {o.label}", c, bold=is_sel)
                elif kind == "action":
                    if data == "build":
                        c = self.C_ON if is_sel else self.C_ON
                    else:
                        c = self.C_HL if is_sel else self.C_DIM
                    self.put(ry, 3, text, c, bold=is_sel)

            self.status_bar(" Up/Down: navigate  |  Space: toggle  |  Left/Right: cycle  |  Enter: select  |  q: back")
            self.scr.refresh()

            k = self.scr.getch()
            ci = selectable[sel]
            ck = rows[ci][0]
            cd = rows[ci][1]

            if k == curses.KEY_UP:
                sel = (sel - 1) % len(selectable)
            elif k == curses.KEY_DOWN:
                sel = (sel + 1) % len(selectable)
            elif k in (ord(" "), curses.KEY_RIGHT, curses.KEY_LEFT):
                d = -1 if k == curses.KEY_LEFT else 1
                if ck == "setting":
                    if cd == "compiler":
                        self.compiler_idx = (self.compiler_idx + d) % len(self.compilers)
                    elif cd == "build_type":
                        self.bt_idx = (self.bt_idx + d) % len(BUILD_TYPES)
                    elif cd == "cores":
                        self.cores = max(1, min(detect_cores(), self.cores + d * 4))
                    elif cd == "clean":
                        self.clean = not self.clean
                elif ck == "option":
                    cd.enabled = not cd.enabled
            elif k in (10, 13, curses.KEY_ENTER):
                if ck == "action" and cd == "build":
                    return "build"
                elif ck == "action" and cd == "back":
                    return "main"
            elif k == ord("q"):
                return "main"

    # ── Presets ──

    def page_presets(self):
        names = list(PRESETS.keys())
        sel = 0
        while True:
            self.scr.erase()
            h, w = self.scr.getmaxyx()
            y = 1
            self.center(y, " Quick Presets ", self.C_LOGO, bold=True)
            y += 2
            self.hline(y, "Select a preset to apply and build")
            y += 2

            for i, name in enumerate(names):
                p = PRESETS[name]
                is_sel = i == sel
                marker = f" {BULLET} " if is_sel else "   "
                c = self.C_HL if is_sel else 0
                self.put(y, 3, marker, self.C_HL if is_sel else self.C_DIM, bold=is_sel)
                self.put(y, 6, name, c, bold=is_sel)
                y += 1
                desc = p.get("desc", "")
                tags = f"[{p['build_type']}] [{p['compiler']}]"
                self.put(y, 10, f"{tags}  {desc}", self.C_DIM)
                y += 2

            self.status_bar(" Up/Down: navigate  |  Enter: apply & build  |  q: back")
            self.scr.refresh()
            k = self.scr.getch()
            if k == curses.KEY_UP:
                sel = (sel - 1) % len(names)
            elif k == curses.KEY_DOWN:
                sel = (sel + 1) % len(names)
            elif k in (10, 13, curses.KEY_ENTER):
                self._apply_preset(names[sel])
                return "build"
            elif k == ord("q"):
                return "main"

    def _apply_preset(self, name):
        p = PRESETS[name]
        self.bt_idx = BUILD_TYPES.index(p["build_type"])
        for i, (cc, _, _) in enumerate(self.compilers):
            if cc == p["compiler"]:
                self.compiler_idx = i
                break
        for o in self.options:
            if o.key in p["options"]:
                o.enabled = p["options"][o.key]

    # ── Build ──

    def page_build(self):
        self.scr.erase()
        h, w = self.scr.getmaxyx()

        y = self.draw_logo(0)
        self.hline(y)
        y += 1

        cc, cxx, clabel = self.compilers[self.compiler_idx]
        bt = BUILD_TYPES[self.bt_idx]

        cmake_args = [
            "cmake", "..", "-G", "Unix Makefiles",
            f"-DCMAKE_BUILD_TYPE={bt}",
            f"-DCMAKE_C_COMPILER={cc}",
            f"-DCMAKE_CXX_COMPILER={cxx}",
        ]
        for o in self.options:
            cmake_args.append(f"-D{o.cmake_flag}={'ON' if o.enabled else 'OFF'}")

        enabled = [o.key for o in self.options if o.enabled]
        self.put(y, 4, f"Compiler:  {clabel}", self.C_ACCENT)
        y += 1
        self.put(y, 4, f"Type:      {bt}   Cores: {self.cores}   Clean: {'Y' if self.clean else 'N'}", self.C_ACCENT)
        y += 1
        flags_str = ", ".join(enabled) if enabled else "(none)"
        self.put(y, 4, f"Flags:     {flags_str[:w-16]}", self.C_DIM)
        y += 2

        self.scr.refresh()
        t0 = time.time()

        # Phase 1: Clean
        if self.clean and os.path.exists(BUILD_DIR):
            self.put(y, 4, "Cleaning...", self.C_HL)
            self.scr.refresh()
            shutil.rmtree(BUILD_DIR, ignore_errors=True)
            self.put(y, 4, "Cleaning...done", self.C_ON)
            self.scr.refresh()
        os.makedirs(BUILD_DIR, exist_ok=True)

        # Phase 2: CMake
        y += 1
        self.put(y, 4, "CMake configuring...", self.C_HL)
        self.scr.refresh()
        res = subprocess.run(cmake_args, cwd=BUILD_DIR,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res.returncode != 0:
            self.put(y, 4, "CMake FAILED", self.C_ERR, bold=True)
            self._show_log(y + 2, res.stdout)
            return
        self.put(y, 4, "CMake configuring...done", self.C_ON)

        # Phase 3: Make
        y += 2
        bar_y = y
        info_y = y + 1
        file_y = y + 2
        bar_w = min(w - 16, 60)

        self._draw_bar(bar_y, 4, bar_w, 0)
        self.scr.refresh()

        proc = subprocess.Popen(
            ["make", f"-j{self.cores}"], cwd=BUILD_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )

        pct = 0
        errbuf = []
        for line in proc.stdout:
            line = line.rstrip()
            errbuf.append(line)
            if len(errbuf) > 200:
                errbuf.pop(0)
            m = re.match(r'\[\s*(\d+)%\]', line)
            if m:
                p = int(m.group(1))
                if p > pct:
                    pct = p
                    self._draw_bar(bar_y, 4, bar_w, pct)
            if "Building" in line or "Linking" in line:
                parts = line.split()
                fname = parts[-1] if parts else ""
                maxl = w - 8
                if len(fname) > maxl:
                    fname = "..." + fname[-(maxl - 3):]
                self.put(file_y, 4, fname.ljust(maxl), self.C_DIM)
            elapsed = time.time() - t0
            self.put(info_y, 4, f"Elapsed: {elapsed:.0f}s", self.C_HL)
            self.scr.refresh()

        proc.wait()
        elapsed = time.time() - t0

        self.put(file_y, 4, " " * (w - 8))

        if proc.returncode == 0:
            self._draw_bar(bar_y, 4, bar_w, 100)
            ry = file_y + 1
            self.hline(ry)
            ry += 1
            self.center(ry, " BUILD SUCCESSFUL ", self.C_OK, bold=True)
            ry += 2
            self.put(ry, 6, f"Time:     {elapsed:.1f}s", self.C_ON)
            ry += 1
            enc = os.path.join(BIN_DIR, bt, "SvtAv1EncApp")
            lib = os.path.join(BIN_DIR, bt, "libSvtAv1Enc.so")
            self.put(ry, 6, f"Encoder:  {enc}", self.C_ACCENT)
            ry += 1
            self.put(ry, 6, f"Library:  {lib}", self.C_ACCENT)
            ry += 2
            self.put(ry, 6, "Press any key to return...", self.C_HL)
        else:
            self._draw_bar(bar_y, 4, bar_w, pct)
            ry = file_y + 1
            self.center(ry, " BUILD FAILED ", self.C_ERR, bold=True)
            self._show_log(ry + 2, "\n".join(errbuf[-20:]))

        self.scr.refresh()
        self.scr.getch()

    def _draw_bar(self, y, x, width, pct):
        filled = int(width * pct / 100)
        empty = width - filled
        self.put(y, x, "[", 0, bold=True)
        self.put(y, x + 1, "\u2588" * filled, self.C_ON, bold=True)
        self.put(y, x + 1 + filled, "\u2591" * empty, self.C_DIM)
        self.put(y, x + 1 + width, f"] {pct:3d}%", 0, bold=True)

    def _show_log(self, y, text):
        h, w = self.scr.getmaxyx()
        lines = text.split("\n") if isinstance(text, str) else [text]
        for i, line in enumerate(lines[-min(15, h - y - 3):]):
            self.put(y + i, 6, line[:w - 8], self.C_RED)
        self.put(y + min(15, h - y - 3) + 1, 6, "Press any key to return...", self.C_HL)
        self.scr.refresh()
        self.scr.getch()


def main(stdscr):
    TUI(stdscr).run()

if __name__ == "__main__":
    os.environ.setdefault("ESCDELAY", "25")
    curses.wrapper(main)
