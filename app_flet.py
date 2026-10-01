from __future__ import annotations

import os
import asyncio
import time
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import flet as ft
try:
    import flet_charts as fch
except Exception:
    fch = None

from gui import chart_events as ce
from gui.image_utils import draw_detections, get_image_info, render_detections_png, validate_image_path
from gui.model_utils import auto_detect_model, load_yolo_model, run_inference
from gui.report_utils import build_report, filter_report, save_report_csv, save_report_json
from gui.state_store import load_state, update_state
from gui.system_info import cpu_name
from gui.theme import PALETTES

LOGGER = logging.getLogger("yolo_gui_dashboard")
if not LOGGER.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(page: ft.Page) -> None:
    page.title = "YOLO Inference Dashboard"
    page.window.min_width = 1280
    page.window.min_height = 820
    page.padding = 18
    page.scroll = ft.ScrollMode.AUTO
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    project_root = Path(__file__).resolve().parent
    artifacts_dir = project_root / "gui_outputs"
    artifacts_dir.mkdir(exist_ok=True)
    state_path = artifacts_dir / "gui_state.json"
    saved_state = load_state(state_path)

    theme_mode = saved_state.get("theme", "dark")
    if theme_mode not in PALETTES:
        theme_mode = "dark"
    palette = PALETTES[theme_mode].copy()

    # The slider can go down to the threshold inference runs with; anything below it
    # would have been discarded by the model already.
    conf_slider_min = 0.05
    conf_slider_max = 0.95
    inference_conf = conf_slider_min
    default_iou = 0.45

    state: dict[str, Any] = {
        "theme": theme_mode,
        "model_path": auto_detect_model(project_root, preferred=saved_state.get("model_path")),
        "model": None,
        "images": [],
        "active_index": -1,
        "conf_filter": 0.25,
        "ui_state": "idle",
        "highlight_token": 0,
        "hover_last_ts": 0.0,
        "hover_last_text": "",
        "hover_last_index": {"conf": -1, "class": -1, "scatter": -1},
        "log_last_ts": {},
        "scatter_points": [],
        "scatter_range": {"min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 1.0},
        "scatter_host_size": {"w": 0.0, "h": 0.0},
    }
    zoom_state: dict[str, float] = {"source": 1.0, "result": 1.0}
    zoom_min = 0.5
    zoom_max = 4.0
    zoom_step = 0.15
    max_content_width = 1720
    input_text_size = 13

    # Native file dialogs; the service registers itself with the page on creation.
    file_picker = ft.FilePicker()

    def color(key: str) -> str:
        return palette[key]

    def log_msg(message: str, level: str = "info", throttle_key: str | None = None, min_interval_sec: float = 0.35) -> None:
        if throttle_key:
            now = time.monotonic()
            log_last = state.get("log_last_ts")
            if not isinstance(log_last, dict):
                log_last = {}
                state["log_last_ts"] = log_last
            prev = float(log_last.get(throttle_key, 0.0))
            if now - prev < min_interval_sec:
                return
            log_last[throttle_key] = now

        if level == "debug":
            LOGGER.debug(message)
        elif level == "warning":
            LOGGER.warning(message)
        elif level == "error":
            LOGGER.error(message)
        else:
            LOGGER.info(message)

    device_label = f"CPU ({cpu_name()})"

    def toast(message: str, kind: str = "accent") -> None:
        bg = color(kind if kind in ("accent", "success", "warning", "error") else "accent")
        page.show_dialog(ft.SnackBar(ft.Text(message, color=color("text_primary")), bgcolor=bg))

    def _apply_detection_card_visual(ui_state: str) -> None:
        if ui_state == "running":
            det_card.border = ft.Border.all(2, color("accent"))
            det_card.shadow = [ft.BoxShadow(blur_radius=18, spread_radius=0, color=ft.Colors.with_opacity(0.35, color("accent")))]
            det_card.opacity = 1.0
        elif ui_state == "success":
            det_card.border = ft.Border.all(2, color("success"))
            det_card.shadow = [ft.BoxShadow(blur_radius=20, spread_radius=1, color=ft.Colors.with_opacity(0.28, color("success")))]
            det_card.opacity = 1.0
        elif ui_state == "empty":
            det_card.border = ft.Border.all(2, color("warning"))
            det_card.shadow = [ft.BoxShadow(blur_radius=16, spread_radius=0, color=ft.Colors.with_opacity(0.22, color("warning")))]
            det_card.opacity = 1.0
        elif ui_state == "error":
            det_card.border = ft.Border.all(2, color("error"))
            det_card.shadow = [ft.BoxShadow(blur_radius=18, spread_radius=0, color=ft.Colors.with_opacity(0.22, color("error")))]
            det_card.opacity = 1.0
        else:  # idle
            det_card.border = ft.Border.all(1, color("border"))
            det_card.shadow = None
            det_card.opacity = 0.98

    async def _clear_success_highlight_later(token: int, delay_sec: float = 1.6) -> None:
        await asyncio.sleep(delay_sec)
        if state.get("highlight_token") != token:
            return
        if state.get("ui_state") == "success":
            _apply_detection_card_visual("idle")
            page.update()

    def set_ui_state(ui_state: str, status_text: str = "") -> None:
        state["ui_state"] = ui_state

        analyze_btn.disabled = ui_state == "running"
        source_loading_overlay.visible = ui_state == "running"
        result_loading_overlay.visible = ui_state == "running"
        processing_ring.visible = ui_state == "running"
        processing_bar.visible = ui_state == "running"
        processing_strip_host.visible = ui_state == "running"

        if ui_state == "running":
            processing_chip.bgcolor = color("accent")
            processing_chip_icon.icon = ft.Icons.AUTO_MODE
            processing_chip_text.value = status_text or "Analyzing image..."
            processing_strip_label.value = status_text or "Analyzing image..."
            processing_strip.value = None
            analyze_btn.content = "Analyzing..."
            analyze_btn.icon = ft.Icons.HOURGLASS_TOP
            sum_card.opacity = 0.92
            charts_card.opacity = 0.92
        else:
            chip_colors = {"success": "success", "empty": "warning", "error": "error"}
            chip_icons = {"empty": ft.Icons.INFO, "error": ft.Icons.ERROR_OUTLINE}
            default_texts = {"success": "Detection completed", "empty": "No objects detected", "error": "Inference error"}
            text = status_text or default_texts.get(ui_state, "Ready")
            processing_chip.bgcolor = color(chip_colors.get(ui_state, "success"))
            processing_chip_icon.icon = chip_icons.get(ui_state, ft.Icons.CHECK_CIRCLE)
            processing_chip_text.value = text
            processing_strip_label.value = text
            processing_strip.value = 0.0
            analyze_btn.content = "Analyze active image"
            analyze_btn.icon = ft.Icons.PLAY_ARROW
            sum_card.opacity = 1.0
            charts_card.opacity = 1.0

        _apply_detection_card_visual(ui_state)
        state["highlight_token"] = int(state.get("highlight_token", 0)) + 1
        if ui_state == "success":
            page.run_task(_clear_success_highlight_later, state["highlight_token"], 1.6)

    def make_card(content: ft.Control, *, padding: int = 16, expand: bool = False, height: int | None = None) -> ft.Container:
        return ft.Container(
            content=content,
            padding=padding,
            border_radius=14,
            bgcolor=color("card"),
            border=ft.Border.all(1, color("border")),
            expand=expand,
            height=height,
        )

    def load_model_if_needed(force: bool = False) -> None:
        model_path = state["model_path"]
        if not model_path:
            raise FileNotFoundError("Model is not selected.")
        if state["model"] is None or force:
            state["model"] = load_yolo_model(Path(model_path))

    def update_model_widgets() -> None:
        model_path = state["model_path"]
        if model_path and Path(model_path).exists():
            model_status.value = "Model: ready"
            model_status.color = color("success")
            model_path_text.value = str(model_path)
        else:
            model_status.value = "Model: not found"
            model_status.color = color("error")
            model_path_text.value = "Select .pt file manually"

    def clear_result_view(message: str = "No inference yet.") -> None:
        summary_list.controls = [ft.Text(message, color=color("text_secondary"))]
        result_table.rows = []
        detail_log.value = ""
        result_image.src = ""
        result_image.visible = False
        if fch is not None:
            confidence_chart.groups = []
            class_chart.groups = []
            size_conf_chart.spots = []
        set_analytics_visibility(False)
        apply_zoom("result", value=1.0, refresh=False)

    def update_summary(report: dict[str, Any], detections: list[dict[str, Any]]) -> None:
        def kv(label: str, value: str) -> ft.Row:
            return ft.Row(
                [
                    ft.Text(label, weight=ft.FontWeight.W_700, color=color("text_primary")),
                    ft.Text(value, color=color("text_secondary"), selectable=True, expand=True),
                ],
                vertical_alignment=ft.CrossAxisAlignment.START,
            )

        total_objects = len(detections)
        unique_classes = sorted({d["class_name"] for d in detections})
        avg_conf = sum(float(d["confidence"]) for d in detections) / total_objects if total_objects else 0.0
        img = report["image"]
        summary_list.controls = [
            kv("File:", img["filename"]),
            kv("Image size:", f"{img['width']}x{img['height']}"),
            kv("Objects:", str(total_objects)),
            kv("Unique classes:", ", ".join(unique_classes) if unique_classes else "-"),
            kv("Avg confidence:", f"{avg_conf:.4f}"),
            kv("Inference time:", f"{report['summary']['inference_ms']} ms"),
            kv("Conf threshold:", f"{state['conf_filter']:.2f}"),
            kv("Device:", report.get("device", device_label)),
            kv("Model:", report["model_path"]),
        ]

    def detections_for_view(item: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not item:
            return []
        threshold = float(state.get("conf_filter", 0.0))
        return [d for d in item.get("detections", []) if float(d.get("confidence", 0.0)) >= threshold]

    def update_table(detections: list[dict[str, Any]]) -> None:
        rows: list[ft.DataRow] = []
        for d in detections:
            b = d["bbox"]
            c = d["center"]
            rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(d["index"]), size=12)),
                        ft.DataCell(ft.Text(d["class_name"], size=12)),
                        ft.DataCell(ft.Text(f'{d["confidence"]:.4f}', size=12)),
                        ft.DataCell(ft.Text(str(b["x_min"]), size=12)),
                        ft.DataCell(ft.Text(str(b["y_min"]), size=12)),
                        ft.DataCell(ft.Text(str(b["x_max"]), size=12)),
                        ft.DataCell(ft.Text(str(b["y_max"]), size=12)),
                        ft.DataCell(ft.Text(str(d["width"]), size=12)),
                        ft.DataCell(ft.Text(str(d["height"]), size=12)),
                        ft.DataCell(ft.Text(str(c["x"]), size=12)),
                        ft.DataCell(ft.Text(str(c["y"]), size=12)),
                    ]
                )
            )
        result_table.rows = rows

    def update_log(detections: list[dict[str, Any]]) -> None:
        if not detections:
            detail_log.value = "No objects detected."
            return
        lines = []
        for d in detections:
            b = d["bbox"]
            lines.append(
                f"#{d['index']} {d['class_name']} (id={d['class_id']}) conf={d['confidence']:.4f} "
                f"bbox=({b['x_min']},{b['y_min']},{b['x_max']},{b['y_max']}) "
                f"w={d['width']} h={d['height']} area={d['area']} center=({d['center']['x']},{d['center']['y']})"
            )
        detail_log.value = "\n".join(lines)

    def update_chart_view(detections: list[dict[str, Any]]) -> None:
        state["hover_last_index"] = {"conf": -1, "class": -1, "scatter": -1}
        state["scatter_points"] = []
        if not detections:
            set_analytics_visibility(False)
            chart_hover_info.value = "Hover bars or click scatter points to inspect object details."
            return
        if fch is None:
            charts_empty.value = "Install 'flet-charts' to enable built-in analytics charts."
            set_analytics_visibility(False)
            chart_hover_info.value = "Native charts are unavailable."
            return
        set_analytics_visibility(True)

        # Confidence bar chart
        conf_groups = []
        conf_labels = []
        for i, d in enumerate(detections):
            conf = float(d["confidence"])
            conf_groups.append(
                fch.BarChartGroup(
                    x=i,
                    rods=[
                        fch.BarChartRod(
                            from_y=0,
                            to_y=conf,
                            width=14,
                            color=ft.Colors.RED_300 if conf < 0.8 else ft.Colors.BLUE_300,
                            border_radius=2,
                        )
                    ],
                )
            )
            conf_labels.append(fch.ChartAxisLabel(value=i, label=ft.Text(str(d["index"]), size=11)))

        confidence_chart.groups = conf_groups
        confidence_chart.bottom_axis = fch.ChartAxis(labels=conf_labels, label_size=30)
        confidence_chart.max_y = 1.0
        confidence_chart.min_y = 0.0

        # Class count bar chart
        class_counts = Counter(d["class_name"] for d in detections)
        cls_items = sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        cls_groups = []
        cls_labels = []
        for i, (name, cnt) in enumerate(cls_items):
            cls_groups.append(
                fch.BarChartGroup(
                    x=i,
                    rods=[
                        fch.BarChartRod(
                            from_y=0,
                            to_y=float(cnt),
                            width=18,
                            color=ft.Colors.GREEN_300,
                            border_radius=2,
                        )
                    ],
                )
            )
            cls_labels.append(fch.ChartAxisLabel(value=i, label=ft.Text(name, size=10)))

        class_chart.groups = cls_groups
        class_chart.bottom_axis = fch.ChartAxis(labels=cls_labels, label_size=56)
        class_chart.max_y = max(1.0, float(max(class_counts.values())))
        class_chart.min_y = 0.0

        # BBox area vs confidence scatter chart
        unique_classes = sorted({d["class_name"] for d in detections})
        cls_colors = [
            ft.Colors.CYAN_300,
            ft.Colors.PURPLE_300,
            ft.Colors.AMBER_300,
            ft.Colors.LIGHT_GREEN_300,
            ft.Colors.PINK_300,
            ft.Colors.INDIGO_300,
            ft.Colors.ORANGE_300,
            ft.Colors.TEAL_300,
        ]
        color_map = {name: cls_colors[i % len(cls_colors)] for i, name in enumerate(unique_classes)}

        spots = []
        scatter_points: list[dict[str, Any]] = []
        min_x = float("inf")
        max_x = 0.0
        for d in detections:
            area = max(1.0, float(d["area"]))
            x = float(area**0.5)  # sqrt(area): stable, readable scale for UI
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            y = float(d["confidence"])
            spots.append(
                fch.ScatterChartSpot(
                    x=x,
                    y=y,
                    color=color_map[d["class_name"]],
                    radius=4,
                )
            )
            scatter_points.append({"x": x, "y": y, "det": d})

        size_conf_chart.spots = spots
        size_conf_chart.min_x = 0.0 if min_x == float("inf") else max(0.0, min_x - 2.0)
        size_conf_chart.max_x = max_x + 2.0
        size_conf_chart.min_y = 0.0
        size_conf_chart.max_y = 1.0
        state["scatter_points"] = scatter_points
        state["scatter_range"] = {
            "min_x": float(size_conf_chart.min_x or 0.0),
            "max_x": float(size_conf_chart.max_x or 1.0),
            "min_y": float(size_conf_chart.min_y or 0.0),
            "max_y": float(size_conf_chart.max_y or 1.0),
        }
        chart_hover_info.value = f"Showing {len(detections)} detection(s) after threshold."

    def set_analytics_visibility(has_data: bool) -> None:
        confidence_chart_host.visible = has_data and fch is not None
        class_chart_host.visible = has_data and fch is not None
        size_conf_chart_host.visible = has_data and fch is not None
        size_conf_chart_layer.visible = has_data and fch is not None
        chart_hover_info.visible = has_data and fch is not None
        chart_hover_info_box.visible = has_data and fch is not None
        charts_empty.visible = not has_data
        charts_card.height = 1220 if has_data else 220

    def refresh_result_views(item: dict[str, Any]) -> None:
        """Re-render everything that depends on the detections and the confidence slider."""
        detections = detections_for_view(item)
        result_image.src = render_detections_png(item["path"], detections)
        result_image.visible = True
        update_summary(item["report"], detections)
        update_table(detections)
        update_log(detections)
        update_chart_view(detections)

    def set_chart_hover_info(text: str, min_interval_sec: float = 0.12) -> None:
        if not text:
            return
        now = time.monotonic()
        last_ts = float(state.get("hover_last_ts", 0.0))
        last_text = str(state.get("hover_last_text", ""))

        if text == last_text:
            return
        if (now - last_ts) < min_interval_sec:
            return

        state["hover_last_ts"] = now
        state["hover_last_text"] = text
        chart_hover_info.value = text
        try:
            chart_hover_info.update()
        except Exception:
            pass

    def _should_process_hover_index(chart_key: str, idx: int) -> bool:
        last_map = state.get("hover_last_index")
        if not isinstance(last_map, dict):
            return True
        if int(last_map.get(chart_key, -1)) == idx:
            return False
        last_map[chart_key] = idx
        return True

    def on_conf_chart_event(e: Any) -> None:
        try:
            if not ce.is_chart_interaction_event(e):
                return
            detections = detections_for_view(current_item())
            idx = ce.event_index(e, "group_index", "bar_group_index", "index")
            log_msg(f"[conf_chart] event: {ce.event_short_dump(e)}", throttle_key="event_conf", min_interval_sec=0.4)
            if 0 <= idx < len(detections):
                if not _should_process_hover_index("conf", idx):
                    return
                d = detections[idx]
                set_chart_hover_info(
                    f"Confidence chart: #{d['index']} {d['class_name']} | conf={d['confidence']:.3f} "
                    f"| bbox=({d['bbox']['x_min']},{d['bbox']['y_min']},{d['bbox']['x_max']},{d['bbox']['y_max']})"
                )
        except Exception:
            LOGGER.exception("[conf_chart] handler failed")

    def on_class_chart_event(e: Any) -> None:
        try:
            if not ce.is_chart_interaction_event(e):
                return
            detections = detections_for_view(current_item())
            if not detections:
                return
            class_counts = Counter(d["class_name"] for d in detections)
            cls_items = sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            idx = ce.event_index(e, "group_index", "bar_group_index", "index")
            log_msg(f"[class_chart] event: {ce.event_short_dump(e)}", throttle_key="event_class", min_interval_sec=0.4)
            if 0 <= idx < len(cls_items):
                if not _should_process_hover_index("class", idx):
                    return
                name, cnt = cls_items[idx]
                set_chart_hover_info(f"Class chart: {name} | count={cnt}")
        except Exception:
            LOGGER.exception("[class_chart] handler failed")

    def on_scatter_host_size_change(e: ft.ControlEvent) -> None:
        w = float(getattr(e, "width", 0.0) or 0.0)
        h = float(getattr(e, "height", 0.0) or 0.0)
        state["scatter_host_size"] = {"w": w, "h": h}

    def on_scatter_host_tap(e: ft.TapEvent) -> None:
        try:
            points = state.get("scatter_points", [])
            if not points:
                return
            lp = getattr(e, "local_position", None)
            if lp is None:
                return
            px = float(getattr(lp, "x", 0.0) or 0.0)
            py = float(getattr(lp, "y", 0.0) or 0.0)

            host_size = state.get("scatter_host_size", {})
            host_w = float(host_size.get("w", 0.0) or 0.0)
            host_h = float(host_size.get("h", 0.0) or 0.0)
            if host_w <= 1.0 or host_h <= 1.0:
                return

            # Approximate chart plot area inside the host (axes + container paddings).
            pad_left = 68.0
            pad_right = 16.0
            pad_top = 14.0
            pad_bottom = 54.0
            plot_w = max(1.0, host_w - pad_left - pad_right)
            plot_h = max(1.0, host_h - pad_top - pad_bottom)

            xr = state.get("scatter_range", {})
            min_x = float(xr.get("min_x", 0.0))
            max_x = float(xr.get("max_x", 1.0))
            min_y = float(xr.get("min_y", 0.0))
            max_y = float(xr.get("max_y", 1.0))
            x_span = max(1e-6, max_x - min_x)
            y_span = max(1e-6, max_y - min_y)

            nx = min(1.0, max(0.0, (px - pad_left) / plot_w))
            ny = min(1.0, max(0.0, (py - pad_top) / plot_h))
            click_x = min_x + nx * x_span
            click_y = max_y - ny * y_span

            nearest = None
            nearest_dist = float("inf")
            for p in points:
                dx = (float(p["x"]) - click_x) / x_span
                dy = (float(p["y"]) - click_y) / y_span
                dist = dx * dx + dy * dy
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = p

            if nearest is None:
                return
            d = nearest["det"]
            set_chart_hover_info(
                f"Area/conf chart: #{d['index']} {d['class_name']} | area={d['area']} | conf={d['confidence']:.3f}"
            )
            log_msg(
                f"[scatter_tap] px={px:.1f}, py={py:.1f}, nearest=#{d['index']} class={d['class_name']} "
                f"conf={d['confidence']:.3f} area={d['area']} dist={nearest_dist:.4f}",
                throttle_key="scatter_tap",
                min_interval_sec=0.08,
            )
        except Exception:
            LOGGER.exception("[scatter_tap] handler failed")

    def clamp_zoom(value: float) -> float:
        return max(zoom_min, min(zoom_max, value))

    def apply_zoom(target: str, value: float | None = None, delta: float = 0.0, refresh: bool = True) -> None:
        current = zoom_state[target]
        new_value = value if value is not None else current + delta
        zoom = clamp_zoom(new_value)
        zoom_state[target] = zoom
        control = source_image if target == "source" else result_image
        control.scale = zoom
        (source_zoom_text if target == "source" else result_zoom_text).value = f"{zoom:.2f}x"
        if refresh:
            page.update()

    def wheel_zoom_handler(target: str):
        def _handler(e: ft.ScrollEvent) -> None:
            delta_y = getattr(getattr(e, "scroll_delta", None), "y", 0.0) or 0.0
            step = zoom_step if delta_y < 0 else -zoom_step
            apply_zoom(target, delta=step, refresh=True)

        return _handler

    def zoom_toolbar(target: str, zoom_text: ft.Text) -> ft.Row:
        return ft.Row(
            [
                ft.OutlinedButton("Zoom -", icon=ft.Icons.REMOVE, on_click=lambda _: apply_zoom(target, delta=-zoom_step)),
                ft.OutlinedButton("Zoom +", icon=ft.Icons.ADD, on_click=lambda _: apply_zoom(target, delta=zoom_step)),
                ft.OutlinedButton("Reset", icon=ft.Icons.FILTER_CENTER_FOCUS, on_click=lambda _: apply_zoom(target, value=1.0)),
                zoom_text,
            ],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def image_surface(image_control: ft.Image, target: str) -> ft.Control:
        base = ft.Container(
            content=image_control,
            expand=True,
            bgcolor=ft.Colors.TRANSPARENT,
            border_radius=10,
            padding=8,
            alignment=ft.Alignment.CENTER,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )
        viewer = ft.InteractiveViewer(
            content=base,
            min_scale=zoom_min,
            max_scale=zoom_max,
            scale_enabled=False,
            pan_enabled=True,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            boundary_margin=ft.Margin.all(12),
        )
        return ft.GestureDetector(
            content=viewer,
            trackpad_scroll_causes_scale=True,
            on_scroll=wheel_zoom_handler(target),
        )

    def current_item() -> dict[str, Any] | None:
        idx = state["active_index"]
        items = state["images"]
        if 0 <= idx < len(items):
            return items[idx]
        return None

    def update_navigation_state() -> None:
        total = len(state["images"])
        idx = state["active_index"]
        if total == 0 or idx < 0:
            image_counter.value = "0 / 0"
            prev_btn.disabled = True
            next_btn.disabled = True
        else:
            image_counter.value = f"{idx + 1} / {total}"
            prev_btn.disabled = idx <= 0
            next_btn.disabled = idx >= total - 1

    def render_active_image() -> None:
        item = current_item()
        if not item:
            loaded_value.value = "No image selected"
            source_image.src = ""
            source_image.visible = False
            clear_result_view("No image selected.")
            update_navigation_state()
            page.update()
            return

        path: Path = item["path"]
        loaded_value.value = path.name
        source_image.src = str(path)
        source_image.visible = True
        apply_zoom("source", value=1.0, refresh=False)

        if item["report"]:
            refresh_result_views(item)
        else:
            clear_result_view("Ready to run inference for selected image.")

        update_navigation_state()
        page.update()

    def add_images(paths: list[Path]) -> None:
        if not paths:
            return
        existing = {str(i["path"]).lower() for i in state["images"]}
        added = 0
        first_new_index: int | None = None
        for path in paths:
            try:
                validate_image_path(path)
            except (OSError, ValueError):
                continue
            key = str(path).lower()
            if key in existing:
                continue
            existing.add(key)
            state["images"].append(
                {
                    "path": path,
                    "annotated_path": None,
                    "report": None,
                    "detections": [],
                }
            )
            if first_new_index is None:
                first_new_index = len(state["images"]) - 1
            added += 1

        if first_new_index is not None:
            state["active_index"] = first_new_index

        if added == 0:
            toast("No new valid images were added.", "warning")
        else:
            toast(f"Added {added} image(s).", "success")
        render_active_image()

    async def pick_images_click(_: ft.ControlEvent) -> None:
        try:
            files = await file_picker.pick_files(
                dialog_title="Select images",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["png", "jpg", "jpeg"],
                allow_multiple=True,
            )
            add_images([Path(f.path) for f in files or [] if f.path])
        except Exception as ex:
            toast(f"Image pick error: {ex}", "error")

    def go_prev(_: ft.ControlEvent) -> None:
        if state["active_index"] > 0:
            state["active_index"] -= 1
            render_active_image()

    def go_next(_: ft.ControlEvent) -> None:
        if state["active_index"] < len(state["images"]) - 1:
            state["active_index"] += 1
            render_active_image()

    def on_conf_threshold_change(e: ft.ControlEvent) -> None:
        state["conf_filter"] = round(float(e.control.value), 2)
        conf_threshold_value.value = f"{state['conf_filter']:.2f}"
        item = current_item()
        if item and item.get("report"):
            refresh_result_views(item)
        page.update()

    def set_selected_model(model_path: Path) -> None:
        if model_path.suffix.lower() != ".pt":
            raise ValueError("Please select a .pt model file.")
        state["model_path"] = model_path
        state["model"] = None
        update_state(state_path, model_path=str(model_path))
        update_model_widgets()
        toast("Model selected", "success")
        page.update()

    async def pick_model_click(_: ft.ControlEvent) -> None:
        try:
            files = await file_picker.pick_files(
                dialog_title="Select YOLO model",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["pt"],
            )
            if files and files[0].path:
                set_selected_model(Path(files[0].path))
        except Exception as ex:
            toast(f"Model pick error: {ex}", "error")

    def run_analysis(_: ft.ControlEvent) -> None:
        try:
            item = current_item()
            if not item:
                raise ValueError("Select at least one image first.")
            if not state["model_path"]:
                raise FileNotFoundError("Model not found. Choose model manually.")

            set_ui_state("running", "Running YOLO inference...")
            page.update()
            t0 = time.perf_counter()
            log_msg(f"[inference] start image={item['path']}")

            load_model_if_needed()
            image_path: Path = item["path"]
            infer = run_inference(
                model=state["model"],
                image_path=image_path,
                conf_thres=inference_conf,
                iou_thres=default_iou,
                device="cpu",
            )
            detections = infer["detections"]

            item["detections"] = detections
            item["report"] = build_report(
                image_info=get_image_info(image_path),
                model_path=Path(state["model_path"]),
                detections=detections,
                inference_ms=infer["elapsed_ms"],
                conf_thres=inference_conf,
                iou_thres=default_iou,
                device=device_label,
            )
            visible = detections_for_view(item)
            item["annotated_path"] = draw_detections(image_path, visible, out_dir=artifacts_dir / "annotated")

            render_active_image()
            apply_zoom("result", value=1.0, refresh=False)
            if visible:
                set_ui_state("success", f"Detection completed - Found {len(visible)} object(s)")
                toast(f"Inference done. Objects: {len(visible)}", "success")
            else:
                set_ui_state("empty", "No objects detected")
                toast("No objects detected.", "warning")
            elapsed_total = (time.perf_counter() - t0) * 1000.0
            log_msg(
                f"[inference] done image={item['path'].name} det={len(visible)} (raw={len(detections)}) "
                f"model_ms={infer['elapsed_ms']:.1f} total_ms={elapsed_total:.1f}"
            )
        except Exception as ex:
            set_ui_state("error", "Inference error")
            toast(f"Inference error: {ex}", "error")
            LOGGER.exception("[inference] failed")
        finally:
            page.update()

    async def save_report_click(fmt: str) -> None:
        item = current_item()
        if not item or not item["report"]:
            toast("Run inference for active image before saving report.", "warning")
            return
        try:
            selected = await file_picker.save_file(
                dialog_title=f"Save {fmt.upper()} report",
                file_name=f"{item['path'].stem}_report.{fmt}",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=[fmt],
            )
            if not selected:
                return
            path = Path(selected)
            if path.suffix.lower() != f".{fmt}":
                path = path.with_name(f"{path.name}.{fmt}")
            # Export exactly what is on screen: detections above the slider threshold.
            report = filter_report(item["report"], float(state["conf_filter"]))
            if fmt == "json":
                save_report_json(report, path)
            else:
                save_report_csv(report, path)
            toast(f"{fmt.upper()} saved: {path}", "success")
        except Exception as ex:
            toast(str(ex), "error")

    async def save_json_click(_: ft.ControlEvent) -> None:
        await save_report_click("json")

    async def save_csv_click(_: ft.ControlEvent) -> None:
        await save_report_click("csv")

    def open_results_folder(_: ft.ControlEvent) -> None:
        folder = artifacts_dir.resolve()
        if os.name == "nt":
            os.startfile(str(folder))
        else:
            toast(str(folder), "warning")

    def apply_theme(mode: str, refresh: bool = True) -> None:
        state["theme"] = mode
        palette.clear()
        palette.update(PALETTES[mode])
        page.theme_mode = ft.ThemeMode.DARK if mode == "dark" else ft.ThemeMode.LIGHT
        page.theme = ft.Theme(color_scheme_seed=color("accent"))
        page.bgcolor = color("bg")
        update_state(state_path, theme=mode)

        for card in (header_card, input_card, src_card, det_card, sum_card, charts_card, table_card, log_card):
            card.bgcolor = color("card")
            card.border = ft.Border.all(1, color("border"))

        for text in (
            title,
            loaded_label,
            device_label_text,
            conf_threshold_label,
            summary_title,
            charts_title,
            chart_hover_info,
            table_title,
            log_title,
            source_title,
            result_title,
            input_title,
        ):
            text.color = color("text_primary")
        for text in (
            subtitle,
            model_path_text,
            source_zoom_text,
            result_zoom_text,
            image_counter,
            loaded_value,
            device_value,
            conf_threshold_value,
            charts_empty,
        ):
            text.color = color("text_secondary")

        for panel in (
            source_canvas,
            result_canvas,
            table_area,
            device_row,
            loaded_row,
            confidence_chart_host,
            class_chart_host,
            size_conf_chart_host,
            chart_hover_info_box,
        ):
            panel.bgcolor = color("panel")
        detail_log.border_color = color("accent")

        choose_model_btn.style = ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=10),
            bgcolor=color("panel"),
            color=color("text_primary"),
        )
        analyze_btn.style = ft.ButtonStyle(
            padding=20,
            shape=ft.RoundedRectangleBorder(radius=12),
            bgcolor=color("accent"),
            color="#FFFFFF",
            text_style=ft.TextStyle(size=16, weight=ft.FontWeight.W_700),
        )
        processing_ring.color = color("accent")
        processing_bar.color = color("accent")
        processing_bar.bgcolor = color("panel")
        processing_strip.color = color("accent")
        processing_strip.bgcolor = color("panel")
        processing_strip_host.bgcolor = color("card")
        processing_strip_host.border = ft.Border.all(1, color("border"))
        processing_strip_label.color = color("text_secondary")
        processing_chip_icon.color = color("text_primary")
        processing_chip_text.color = color("text_primary")
        theme_switch.thumb_color = color("accent")
        set_ui_state(state.get("ui_state", "idle"), processing_chip_text.value)

        item = current_item()
        if item and item.get("report"):
            refresh_result_views(item)
        else:
            clear_result_view("No image selected." if not item else "Ready to run inference for selected image.")

        update_model_widgets()
        if refresh:
            page.update()

    def on_theme_toggle(_: ft.ControlEvent) -> None:
        mode = "light" if theme_switch.value else "dark"
        apply_theme(mode, refresh=True)

    # Header
    title = ft.Text("YOLO Inference Dashboard", size=28, weight=ft.FontWeight.W_700)
    subtitle = ft.Text("Analyze images with trained YOLO model", size=14)
    model_status = ft.Text("", size=14, weight=ft.FontWeight.W_600)
    model_path_text = ft.Text("", size=12, selectable=True)
    theme_switch = ft.Switch(label="Light theme", value=(state["theme"] == "light"), on_change=on_theme_toggle)

    choose_model_btn = ft.Button(
        "Choose model (.pt)",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=pick_model_click,
    )
    header_card = make_card(
        ft.Row(
            [
                ft.Column([title, subtitle], spacing=4, expand=True),
                ft.Column([model_status, model_path_text], spacing=4, expand=True),
                ft.Row([theme_switch, choose_model_btn], spacing=10),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )
    )

    source_image = ft.Image(src="", fit=ft.BoxFit.CONTAIN, border_radius=10, visible=False)
    result_image = ft.Image(src="", fit=ft.BoxFit.CONTAIN, border_radius=10, visible=False)
    source_zoom_text = ft.Text("1.00x", size=12)
    result_zoom_text = ft.Text("1.00x", size=12)
    image_counter = ft.Text("0 / 0", size=input_text_size)

    loaded_label = ft.Text("Loaded:", weight=ft.FontWeight.W_700, size=input_text_size)
    loaded_value = ft.Text("No image selected", size=input_text_size, selectable=True, expand=True)
    loaded_row = ft.Container(
        ft.Row([loaded_label, loaded_value], spacing=8),
        bgcolor=color("panel"),
        border_radius=8,
        padding=8,
    )

    device_label_text = ft.Text("Device:", weight=ft.FontWeight.W_700, size=input_text_size)
    device_value = ft.Text(device_label, size=input_text_size, selectable=True, expand=True)
    device_row = ft.Container(
        ft.Row([device_label_text, device_value], spacing=8),
        bgcolor=color("panel"),
        border_radius=8,
        padding=8,
    )

    processing_ring = ft.ProgressRing(visible=False, width=18, height=18, stroke_width=2.5)
    processing_bar = ft.ProgressBar(visible=False, width=220)
    processing_strip = ft.ProgressBar(value=0.0, expand=True)
    processing_strip_label = ft.Text("Ready", size=12)
    processing_strip_host = ft.Container(
        content=ft.Row([processing_strip_label, processing_strip], spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        visible=False,
        border_radius=10,
        padding=ft.Padding.symmetric(horizontal=12, vertical=8),
    )
    processing_chip_icon = ft.Icon(ft.Icons.CHECK_CIRCLE, size=16)
    processing_chip_text = ft.Text("Ready", size=input_text_size, weight=ft.FontWeight.W_600)
    processing_chip = ft.Container(
        content=ft.Row([processing_chip_icon, processing_chip_text], spacing=8, tight=True),
        bgcolor=color("success"),
        border_radius=999,
        padding=ft.Padding.symmetric(horizontal=10, vertical=6),
    )
    conf_threshold_label = ft.Text("Confidence threshold", size=input_text_size, weight=ft.FontWeight.W_700)
    conf_threshold_value = ft.Text(f"{state['conf_filter']:.2f}", size=input_text_size)
    conf_threshold_slider = ft.Slider(
        min=conf_slider_min,
        max=conf_slider_max,
        divisions=round((conf_slider_max - conf_slider_min) / 0.05),
        value=state["conf_filter"],
        on_change=on_conf_threshold_change,
    )
    analyze_btn = ft.Button("Analyze active image", icon=ft.Icons.PLAY_ARROW, on_click=run_analysis)
    prev_btn = ft.OutlinedButton("Prev", icon=ft.Icons.ARROW_BACK, on_click=go_prev)
    next_btn = ft.OutlinedButton("Next", icon=ft.Icons.ARROW_FORWARD, on_click=go_next)

    input_title = ft.Text("Input", size=18, weight=ft.FontWeight.W_600)
    input_card = make_card(
        ft.Column(
            [
                input_title,
                ft.Row([ft.Button("Upload image(s)", icon=ft.Icons.UPLOAD_FILE, on_click=pick_images_click), processing_ring]),
                ft.Row([prev_btn, next_btn, image_counter], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Row([processing_chip], wrap=True),
                processing_bar,
                ft.Row([conf_threshold_label, conf_threshold_value], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                conf_threshold_slider,
                loaded_row,
                device_row,
                analyze_btn,
                ft.Row(
                    [
                        ft.OutlinedButton("Save JSON", icon=ft.Icons.SAVE, on_click=save_json_click),
                        ft.OutlinedButton("Save CSV", icon=ft.Icons.TABLE_ROWS, on_click=save_csv_click),
                    ]
                ),
                ft.OutlinedButton("Open results folder", icon=ft.Icons.FOLDER, on_click=open_results_folder),
            ],
            spacing=10,
        ),
    )

    source_loading_overlay = ft.Container(
        content=ft.Column(
            [
                ft.ProgressRing(width=42, height=42, stroke_width=4),
                ft.Text("Processing...", size=14, weight=ft.FontWeight.W_600),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            tight=True,
            spacing=10,
        ),
        bgcolor="#66000000",
        alignment=ft.Alignment.CENTER,
        border_radius=10,
        visible=False,
        expand=True,
    )
    result_loading_overlay = ft.Container(
        content=ft.Column(
            [
                ft.ProgressRing(width=42, height=42, stroke_width=4),
                ft.Text("Rendering result...", size=14, weight=ft.FontWeight.W_600),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            tight=True,
            spacing=10,
        ),
        bgcolor="#66000000",
        alignment=ft.Alignment.CENTER,
        border_radius=10,
        visible=False,
        expand=True,
    )

    source_canvas = ft.Container(
        content=ft.Stack([image_surface(source_image, "source"), source_loading_overlay], expand=True),
        bgcolor=color("panel"),
        border_radius=10,
        expand=True,
    )
    result_canvas = ft.Container(
        content=ft.Stack([image_surface(result_image, "result"), result_loading_overlay], expand=True),
        bgcolor=color("panel"),
        border_radius=10,
        expand=True,
    )
    source_surface = ft.Container(content=source_canvas, expand=True)
    result_surface = ft.Container(content=result_canvas, expand=True)

    source_title = ft.Text("Source image", size=18, weight=ft.FontWeight.W_600)
    result_title = ft.Text("Detection result", size=18, weight=ft.FontWeight.W_600)
    src_card = make_card(
        ft.Column([source_title, zoom_toolbar("source", source_zoom_text), source_surface], expand=True),
        height=420,
    )
    det_card = make_card(
        ft.Column([result_title, zoom_toolbar("result", result_zoom_text), result_surface], expand=True),
        height=420,
    )
    det_card.animate = ft.Animation(280, ft.AnimationCurve.EASE_OUT)
    det_card.animate_scale = ft.Animation(280, ft.AnimationCurve.EASE_OUT)
    det_card.animate_opacity = ft.Animation(220, ft.AnimationCurve.EASE_OUT)
    det_card.opacity = 0.98

    summary_title = ft.Text("Summary", size=18, weight=ft.FontWeight.W_600)
    summary_list = ft.Column(controls=[ft.Text("No inference yet.")], spacing=8, scroll=ft.ScrollMode.AUTO, expand=True)
    sum_card = make_card(
        ft.Column([summary_title, ft.Container(content=summary_list, expand=True)], expand=True),
        height=420,
    )
    sum_card.animate_opacity = ft.Animation(240, ft.AnimationCurve.EASE_OUT)

    table_title = ft.Text("Detections table", size=18, weight=ft.FontWeight.W_600)
    result_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("#")),
            ft.DataColumn(ft.Text("class")),
            ft.DataColumn(ft.Text("conf")),
            ft.DataColumn(ft.Text("x1")),
            ft.DataColumn(ft.Text("y1")),
            ft.DataColumn(ft.Text("x2")),
            ft.DataColumn(ft.Text("y2")),
            ft.DataColumn(ft.Text("w")),
            ft.DataColumn(ft.Text("h")),
            ft.DataColumn(ft.Text("cx")),
            ft.DataColumn(ft.Text("cy")),
        ],
        rows=[],
        column_spacing=12,
        data_row_min_height=28,
        data_row_max_height=36,
    )
    table_area = ft.Container(
        content=ft.ListView([result_table], expand=True, spacing=0),
        height=230,
        border_radius=10,
        padding=8,
    )
    table_card = make_card(ft.Column([table_title, table_area]), height=330)

    charts_title = ft.Text("Analytics", size=18, weight=ft.FontWeight.W_600)
    chart_hover_info = ft.Text(
        "Hover bars or click scatter points to inspect object details.",
        size=14,
        weight=ft.FontWeight.W_700,
        color=color("text_primary"),
        selectable=True,
    )
    chart_hover_info_box = ft.Container(
        chart_hover_info,
        bgcolor=color("panel"),
        border_radius=10,
        padding=10,
    )
    if fch is not None:
        confidence_chart = fch.BarChart(
            expand=True,
            max_y=1.0,
            min_y=0.0,
            horizontal_grid_lines=fch.ChartGridLines(color=ft.Colors.with_opacity(0.15, ft.Colors.GREY)),
            left_axis=fch.ChartAxis(label_size=42),
            bottom_axis=fch.ChartAxis(label_size=32),
            interactive=True,
            on_event=on_conf_chart_event,
        )
        class_chart = fch.BarChart(
            expand=True,
            min_y=0.0,
            horizontal_grid_lines=fch.ChartGridLines(color=ft.Colors.with_opacity(0.15, ft.Colors.GREY)),
            left_axis=fch.ChartAxis(label_size=42),
            bottom_axis=fch.ChartAxis(label_size=56),
            interactive=True,
            on_event=on_class_chart_event,
        )
        size_conf_chart = fch.ScatterChart(
            expand=True,
            min_y=0.0,
            max_y=1.0,
            horizontal_grid_lines=fch.ChartGridLines(color=ft.Colors.with_opacity(0.15, ft.Colors.GREY)),
            vertical_grid_lines=fch.ChartGridLines(color=ft.Colors.with_opacity(0.12, ft.Colors.GREY)),
            left_axis=fch.ChartAxis(label_size=42),
            bottom_axis=fch.ChartAxis(label_size=44),
            interactive=False,
        )
        charts_empty = ft.Text(
            "Run inference to build charts: confidence, class counts and bbox-size diagnostics.",
            color=color("text_secondary"),
        )
    else:
        confidence_chart = ft.Container(
            content=ft.Text("flet-charts not installed", color=color("warning")),
            alignment=ft.Alignment.CENTER,
            expand=True,
        )
        class_chart = ft.Container(
            content=ft.Text("Install flet-charts to enable native charts", color=color("warning")),
            alignment=ft.Alignment.CENTER,
            expand=True,
        )
        size_conf_chart = ft.Container(
            content=ft.Text("pip install flet-charts", color=color("warning")),
            alignment=ft.Alignment.CENTER,
            expand=True,
        )
        charts_empty = ft.Text(
            "Built-in Flet charts are unavailable until flet-charts is installed.",
            color=color("text_secondary"),
        )
    confidence_chart_host = ft.Container(
        confidence_chart,
        bgcolor=color("panel"),
        border_radius=10,
        padding=10,
        height=340,
        clip_behavior=ft.ClipBehavior.NONE,
        visible=False,
    )
    class_chart_host = ft.Container(
        class_chart,
        bgcolor=color("panel"),
        border_radius=10,
        padding=10,
        height=340,
        clip_behavior=ft.ClipBehavior.NONE,
        visible=False,
    )
    size_conf_chart_host = ft.Container(
        size_conf_chart,
        bgcolor=color("panel"),
        border_radius=10,
        padding=10,
        height=340,
        clip_behavior=ft.ClipBehavior.NONE,
        visible=False,
    )
    size_conf_chart_layer = ft.GestureDetector(
        content=size_conf_chart_host,
        on_tap=on_scatter_host_tap,
        on_size_change=on_scatter_host_size_change,
        visible=False,
    )
    chart_hover_info.visible = False
    charts_card = make_card(
        ft.Column(
            [
                charts_title,
                confidence_chart_host,
                class_chart_host,
                size_conf_chart_layer,
                chart_hover_info_box,
                charts_empty,
            ],
            spacing=12,
        ),
        height=220,
    )
    charts_card.animate_opacity = ft.Animation(240, ft.AnimationCurve.EASE_OUT)

    log_title = ft.Text("Detailed log", size=18, weight=ft.FontWeight.W_600)
    detail_log = ft.TextField(
        value="",
        multiline=True,
        min_lines=10,
        max_lines=20,
        read_only=True,
        expand=True,
        text_size=13,
        border_radius=10,
    )
    log_card = make_card(
        ft.Column([log_title, ft.Container(content=detail_log, expand=True)], expand=True),
        height=360,
    )

    left_panel = ft.Column([input_card], width=380, spacing=14, scroll=ft.ScrollMode.AUTO)
    right_panel = ft.Column(
        [src_card, det_card, sum_card, charts_card, table_card, log_card],
        expand=True,
        spacing=14,
        scroll=ft.ScrollMode.AUTO,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
    )

    main_layout = ft.Column(
        [header_card, processing_strip_host, ft.Row([left_panel, right_panel], expand=True, vertical_alignment=ft.CrossAxisAlignment.START)],
        expand=True,
        spacing=14,
    )

    page.add(ft.Container(content=main_layout, width=max_content_width, expand=True))

    apply_theme(state["theme"], refresh=False)
    update_model_widgets()
    update_navigation_state()
    set_ui_state("idle", "Ready")
    clear_result_view("No image selected.")
    page.update()


if __name__ == "__main__":
    ft.run(main)
