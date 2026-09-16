# -*- coding: utf-8 -*-
"""[wcx2 新增] 资源组键（新）：RJ 简化口径，不含硬件。

组键 = 省|市|原运营商|调度运营商|出省配置|NAT|IPv6|网络质量
质量分档：优≥95 / 良≥90 / 中≥80 / 差<80 / 未知
"""
from __future__ import annotations

UNKNOWN = "未知"
ISP_ALIASES = {
    "ctcc": "电信", "电信": "电信", "chinatelecom": "电信",
    "cucc": "联通", "联通": "联通", "chinaunicom": "联通",
    "cmcc": "移动", "移动": "移动", "chinamobile": "移动",
}
NAT_ALIASES = {
    "full": "full", "fullcone": "full", "nat1": "full", "public": "public",
    "symmetric": "symmetric", "nat4": "symmetric",
    "restricport": "restricPort", "restrictedport": "restricPort",
    "portrestricted": "restricPort", "restricted": "restricted",
}
GROUP_DIMS = (
    "province", "city", "isp", "schedule_isp", "transprov", "nattype", "ipv6", "quality",
)


def cell(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>", "null"}:
        return ""
    return text


def isp_name(value) -> str:
    text = cell(value)
    if not text:
        return UNKNOWN
    return ISP_ALIASES.get(text.lower(), text)


def first_schedule_isp(value, local_isp: str) -> str:
    text = cell(value)
    if not text:
        return isp_name(local_isp)
    token = text.replace("，", ",").replace("|", ",").replace(";", ",").split(",")[0].strip()
    token = token.strip("[]'\" ")
    return isp_name(token) if token else isp_name(local_isp)


def transprov_bin(value) -> str:
    text = cell(value)
    if not text:
        return "本省"
    try:
        number = float(text)
    except ValueError:
        return UNKNOWN
    if number == 0:
        return "本省"
    if number == 100:
        return "出省"
    return UNKNOWN


def nat_name(value) -> str:
    text = cell(value)
    if not text:
        return UNKNOWN
    return NAT_ALIASES.get(text.lower().replace(" ", ""), text)


def ipv6_bin(value) -> str:
    text = cell(value).lower()
    if not text:
        return UNKNOWN
    if text in {"1", "true", "yes", "y", "支持"}:
        return "支持"
    if text in {"0", "false", "no", "n", "不支持"}:
        return "不支持"
    return UNKNOWN


def quality_bin(pct) -> str:
    if pct is None or (isinstance(pct, float) and pct != pct):
        return UNKNOWN
    text = cell(pct)
    if not text:
        return UNKNOWN
    try:
        number = float(text)
    except ValueError:
        return UNKNOWN
    if number >= 95:
        return "优"
    if number >= 90:
        return "良"
    if number >= 80:
        return "中"
    return "差"


def group_fields(row: dict) -> dict[str, str]:
    local_isp = isp_name(row.get("isp"))
    return {
        "province": cell(row.get("province")) or UNKNOWN,
        "city": cell(row.get("city")) or UNKNOWN,
        "isp": local_isp,
        "schedule_isp": first_schedule_isp(row.get("scheduleisps"), local_isp),
        "transprov": transprov_bin(row.get("transprovrate")),
        "nattype": nat_name(row.get("nattype")),
        "ipv6": ipv6_bin(row.get("ipv6")),
        "quality": quality_bin(row.get("quality_pct")),
    }


def group_key(row: dict) -> str:
    fields = group_fields(row)
    return "|".join(fields[name] for name in GROUP_DIMS)
