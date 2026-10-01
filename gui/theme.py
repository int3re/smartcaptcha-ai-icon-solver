from __future__ import annotations

import flet as ft


BG = "#0B1020"
PANEL = "#131A2E"
CARD = "#18223A"
ACCENT = "#4DA3FF"
SUCCESS = "#3DDC97"
WARNING = "#FFB020"
ERROR = "#FF5D73"
TEXT_PRIMARY = "#E8EEF9"
TEXT_SECONDARY = "#9FB0D0"
BORDER = "#263657"


def apply_dark_theme(page: ft.Page) -> None:
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = BG
    page.theme = ft.Theme(color_scheme_seed=ACCENT)


def card(content: ft.Control, *, padding: int = 16, expand: bool = False, height: int | None = None) -> ft.Container:
    return ft.Container(
        content=content,
        padding=padding,
        border_radius=14,
        bgcolor=CARD,
        border=ft.border.all(1, BORDER),
        expand=expand,
        height=height,
    )
