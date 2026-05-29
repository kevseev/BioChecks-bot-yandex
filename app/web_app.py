"""Веб-интерфейс с тем же конвейером LUNA, что и у бота."""

from __future__ import annotations

import base64
import logging
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from app import access_control, pipeline
from app.access_control import ensure_bootstrap
from app.config import settings
from app.state import Flow
from app.url_images import extract_urls, fetch_image_from_url

log = logging.getLogger(__name__)

COOKIE = "biochecks_web_sid"
_WEB_STORE: dict[str, "WebState"] = {}


@dataclass
class WebState:
    flow: Flow = Flow.IDLE
    compare_buffers: list[tuple[bytes, str]] = field(default_factory=list)
    login: str | None = None
    authorized: bool = False


def _norm_ct(ctype: str) -> str:
    c = (ctype or "").split(";")[0].strip().lower()
    if c in ("application/octet-stream", "", "binary/octet-stream"):
        return "image/jpeg"
    return c


def _ensure_web_auth_defaults(st: WebState) -> None:
    if not settings.bot_auth_enabled:
        st.authorized = True
        if not st.login:
            st.login = "web"


def _new_sid() -> str:
    return secrets.token_urlsafe(32)


def _get_or_create_state(request: Request) -> tuple[str, WebState]:
    sid = request.cookies.get(COOKIE)
    if sid and sid in _WEB_STORE:
        st = _WEB_STORE[sid]
        _ensure_web_auth_defaults(st)
        return sid, st
    sid = _new_sid()
    st = WebState()
    _ensure_web_auth_defaults(st)
    _WEB_STORE[sid] = st
    return sid, st


def _attach_cookie(response: Response, sid: str) -> None:
    response.set_cookie(
        key=COOKIE,
        value=sid,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
        path="/",
    )


def _bundle_to_result(bundle: pipeline.BotBundle) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for p in bundle.parts:
        if isinstance(p, pipeline.PipeText):
            items.append({"kind": "text", "text": p.text})
        else:
            items.append(
                {
                    "kind": "image",
                    "filename": p.filename,
                    "caption": p.caption or "",
                    "data_base64": base64.standard_b64encode(p.data).decode("ascii"),
                }
            )
    return {"items": items, "menu_hint": bundle.menu_message}


async def _run_bundle_safe(coro) -> dict[str, Any]:
    try:
        bundle = await coro
        return _bundle_to_result(bundle)
    except Exception as e:
        log.exception("web analyze")
        return {
            "items": [{"kind": "text", "text": f"Ошибка: {e}"[:3000]}],
            "menu_hint": None,
        }


FLOW_API_TO_ENUM: dict[str, Flow] = {
    "idle": Flow.IDLE,
    "wait_compare": Flow.WAIT_COMPARE,
    "wait_attributes": Flow.WAIT_ATTRIBUTES,
    "wait_body_attributes": Flow.WAIT_BODY_ATTRIBUTES,
    "wait_liveness": Flow.WAIT_LIVENESS,
    "wait_deepfake": Flow.WAIT_DEEPFAKE,
    "wait_quality": Flow.WAIT_QUALITY,
    "wait_face_detect": Flow.WAIT_FACE_DETECT,
    "wait_body_detect": Flow.WAIT_BODY_DETECT,
    "wait_image_modification": Flow.WAIT_IMAGE_MODIFICATION,
    "wait_crowd_detect": Flow.WAIT_CROWD_DETECT,
}

FLOW_ENUM_TO_API: dict[Flow, str] = {v: k for k, v in FLOW_API_TO_ENUM.items()}


class LoginBody(BaseModel):
    login: str
    password: str


class FlowBody(BaseModel):
    flow: str


class AdminIssueBody(BaseModel):
    login: str


@asynccontextmanager
async def _lifespan(_: FastAPI):
    try:
        ensure_bootstrap()
    except Exception:
        log.exception("ensure_bootstrap web")
        raise
    yield


app = FastAPI(title="BioChecks Web", lifespan=_lifespan)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    sid, _st = _get_or_create_state(request)
    resp = HTMLResponse(content=INDEX_HTML)
    _attach_cookie(resp, sid)
    return resp


def _require_api_session(request: Request) -> tuple[str, WebState]:
    sid = request.cookies.get(COOKIE)
    if not sid or sid not in _WEB_STORE:
        raise HTTPException(status_code=401, detail="Нет сессии. Откройте страницу заново.")
    st = _WEB_STORE[sid]
    _ensure_web_auth_defaults(st)
    if settings.bot_auth_enabled and not st.authorized:
        raise HTTPException(status_code=401, detail="Требуется вход.")
    return sid, st


@app.get("/api/me")
async def api_me(request: Request) -> JSONResponse:
    sid, st = _get_or_create_state(request)
    _ensure_web_auth_defaults(st)
    admin = bool(st.login and st.login in access_control.admin_logins()) if st.login else False
    body = {
        "auth_enabled": settings.bot_auth_enabled,
        "authorized": st.authorized,
        "login": st.login,
        "is_admin": admin,
        "flow": FLOW_ENUM_TO_API.get(st.flow, "idle"),
        "compare_count": len(st.compare_buffers),
    }
    resp = JSONResponse(body)
    _attach_cookie(resp, sid)
    return resp


@app.post("/api/login")
async def api_login(request: Request, body: LoginBody) -> JSONResponse:
    if not settings.bot_auth_enabled:
        raise HTTPException(status_code=400, detail="Авторизация отключена (BOT_AUTH_ENABLED=false).")
    sid, st = _get_or_create_state(request)
    login_n = access_control.normalize_login(body.login)
    if not login_n:
        raise HTTPException(status_code=400, detail="Пустой логин.")
    if not access_control.user_has_record(login_n):
        raise HTTPException(status_code=403, detail="Доступ не настроен для этого логина.")
    if not access_control.verify_password(login_n, body.password):
        raise HTTPException(status_code=403, detail="Неверный пароль.")
    st.login = login_n
    st.authorized = True
    st.flow = Flow.IDLE
    st.compare_buffers = []
    resp = JSONResponse({"ok": True, "login": login_n})
    _attach_cookie(resp, sid)
    return resp


@app.post("/api/logout")
async def api_logout(request: Request) -> JSONResponse:
    sid, st = _require_api_session(request)
    st.authorized = False
    st.login = None
    st.flow = Flow.IDLE
    st.compare_buffers = []
    resp = JSONResponse({"ok": True})
    _attach_cookie(resp, sid)
    return resp


@app.post("/api/flow")
async def api_flow(request: Request, body: FlowBody) -> JSONResponse:
    _, st = _require_api_session(request)
    key = (body.flow or "").strip().lower()
    if key not in FLOW_API_TO_ENUM:
        raise HTTPException(status_code=400, detail="Неизвестный режим.")
    st.flow = FLOW_API_TO_ENUM[key]
    st.compare_buffers = []
    return JSONResponse({"ok": True, "flow": FLOW_ENUM_TO_API[st.flow]})


@app.post("/api/reset")
async def api_reset(request: Request) -> JSONResponse:
    _, st = _require_api_session(request)
    st.flow = Flow.IDLE
    st.compare_buffers = []
    return JSONResponse({"ok": True, "flow": "idle"})


@app.get("/api/admin/users")
async def api_admin_users(request: Request) -> JSONResponse:
    _, st = _require_api_session(request)
    if not st.login or st.login not in access_control.admin_logins():
        raise HTTPException(status_code=403, detail="Нужны права администратора.")
    return JSONResponse({"users": access_control.list_user_logins()})


@app.post("/api/admin/issue")
async def api_admin_issue(request: Request, body: AdminIssueBody) -> JSONResponse:
    _, st = _require_api_session(request)
    if not st.login or st.login not in access_control.admin_logins():
        raise HTTPException(status_code=403, detail="Нужны права администратора.")
    target = access_control.normalize_login(body.login)
    if not target:
        raise HTTPException(status_code=400, detail="Пустой логин.")
    pw = access_control.generate_password()
    access_control.set_user_password(target, pw)
    return JSONResponse({"login": target, "password": pw})


@app.post("/api/admin/revoke")
async def api_admin_revoke(request: Request, body: AdminIssueBody) -> JSONResponse:
    _, st = _require_api_session(request)
    if not st.login or st.login not in access_control.admin_logins():
        raise HTTPException(status_code=403, detail="Нужны права администратора.")
    target = access_control.normalize_login(body.login)
    if access_control.delete_user(target):
        return JSONResponse({"ok": True, "revoked": target})
    raise HTTPException(status_code=404, detail="Логин не найден.")


async def web_dispatch(
    st: WebState,
    parts: list[tuple[bytes, str]],
    failed_urls: list[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    warnings: list[dict[str, Any]] = []

    if not parts:
        if failed_urls:
            details = "\n".join(f"• {u} — {err[:160]}" for u, err in failed_urls[:5])
            warnings.append(
                {
                    "kind": "text",
                    "text": (
                        "🔗 Не удалось скачать фото по ссылкам.\n"
                        "Проверьте, что ссылка публичная и ведет к файлу изображения.\n\n"
                        f"{details}"
                    )[:5900],
                }
            )
        return warnings, None

    if failed_urls:
        details = "\n".join(f"• {u} — {err[:160]}" for u, err in failed_urls[:5])
        warnings.append(
            {
                "kind": "text",
                "text": (
                    "⚠️ Некоторые ссылки не удалось скачать, обрабатываю только успешные.\n\n" f"{details}"
                )[:5900],
            }
        )

    if st.flow == Flow.WAIT_COMPARE:
        for data, ct in parts:
            if len(st.compare_buffers) >= 2:
                break
            st.compare_buffers.append((data, ct))
        if len(st.compare_buffers) >= 2:
            result = await _run_bundle_safe(
                pipeline.run_compare(st.compare_buffers[:2]),
            )
            st.flow = Flow.IDLE
            st.compare_buffers = []
            return warnings, result
        need = 2 - len(st.compare_buffers)
        msg = f"Принято фото {len(st.compare_buffers)}/2. Пришлите ещё {need}."
        return warnings, {"items": [{"kind": "text", "text": msg}], "menu_hint": None}

    if st.flow == Flow.WAIT_ATTRIBUTES:
        data, ct = parts[0]
        result = await _run_bundle_safe(pipeline.run_attributes(data, ct))
        st.flow = Flow.IDLE
        return warnings, result

    if st.flow == Flow.WAIT_BODY_ATTRIBUTES:
        data, ct = parts[0]
        result = await _run_bundle_safe(pipeline.run_body_attributes(data, ct))
        st.flow = Flow.IDLE
        return warnings, result

    if st.flow == Flow.WAIT_LIVENESS:
        data, ct = parts[0]
        result = await _run_bundle_safe(pipeline.run_liveness(data, ct))
        st.flow = Flow.IDLE
        return warnings, result

    if st.flow == Flow.WAIT_DEEPFAKE:
        data, ct = parts[0]
        result = await _run_bundle_safe(pipeline.run_deepfake(data, ct))
        st.flow = Flow.IDLE
        return warnings, result

    if st.flow == Flow.WAIT_QUALITY:
        data, ct = parts[0]
        result = await _run_bundle_safe(pipeline.run_quality(data, ct))
        st.flow = Flow.IDLE
        return warnings, result

    if st.flow == Flow.WAIT_FACE_DETECT:
        data, ct = parts[0]
        result = await _run_bundle_safe(pipeline.run_face_detect(data, ct))
        st.flow = Flow.IDLE
        return warnings, result

    if st.flow == Flow.WAIT_BODY_DETECT:
        data, ct = parts[0]
        result = await _run_bundle_safe(pipeline.run_body_detect(data, ct))
        st.flow = Flow.IDLE
        return warnings, result

    if st.flow == Flow.WAIT_IMAGE_MODIFICATION:
        data, ct = parts[0]
        result = await _run_bundle_safe(pipeline.run_image_modification(data, ct))
        st.flow = Flow.IDLE
        return warnings, result

    if st.flow == Flow.WAIT_CROWD_DETECT:
        data, ct = parts[0]
        result = await _run_bundle_safe(pipeline.run_crowd_detect(data, ct))
        st.flow = Flow.IDLE
        return warnings, result

    if len(parts) == 2 and st.flow == Flow.IDLE:
        result = await _run_bundle_safe(pipeline.run_compare(parts[:2]))
        return warnings, result

    if len(parts) == 1 and st.flow == Flow.IDLE:
        return warnings, {
            "items": [{"kind": "text", "text": "Выберите режим кнопкой ниже, затем снова отправьте фото."}],
            "menu_hint": None,
        }

    return warnings, {"items": [{"kind": "text", "text": "Не удалось обработать запрос."}], "menu_hint": None}


@app.post("/api/analyze")
async def api_analyze(
    request: Request,
    urls_text: str = Form(""),
    files: list[UploadFile] | None = File(None),
) -> JSONResponse:
    _, st = _require_api_session(request)

    parts: list[tuple[bytes, str]] = []
    failed_urls: list[tuple[str, str]] = []

    for uf in files or []:
        if not uf.filename:
            continue
        raw = await uf.read()
        if not raw:
            continue
        parts.append((raw, _norm_ct(uf.content_type or "image/jpeg")))

    combined = (urls_text or "").strip()
    for url in extract_urls(combined):
        try:
            data, ct, _f = await fetch_image_from_url(url)
            parts.append((data, _norm_ct(ct)))
        except Exception as e:
            log.warning("url image %s: %s", url, e)
            failed_urls.append((url, str(e)))

    warnings, result = await web_dispatch(st, parts, failed_urls)
    return JSONResponse(
        {
            "warnings": warnings,
            "result": result,
            "flow": FLOW_ENUM_TO_API.get(st.flow, "idle"),
            "compare_count": len(st.compare_buffers),
        }
    )


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
  <meta name="theme-color" content="#0f172a"/>
  <title>BioChecks — Luna</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent2: #22c55e;
      --danger: #f87171;
      --radius: 12px;
      --touch: 48px;
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, Ubuntu, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100dvh;
      background: var(--bg);
      color: var(--text);
      padding: max(12px, env(safe-area-inset-top)) max(12px, env(safe-area-inset-right))
        max(16px, env(safe-area-inset-bottom)) max(12px, env(safe-area-inset-left));
      line-height: 1.45;
    }
    h1 { font-size: clamp(1.15rem, 4vw, 1.45rem); margin: 0 0 8px; font-weight: 650; }
    p.lead { margin: 0 0 16px; color: var(--muted); font-size: 0.95rem; }
    .card {
      background: var(--card);
      border-radius: var(--radius);
      padding: 14px;
      margin-bottom: 14px;
      border: 1px solid rgba(148,163,184,0.15);
    }
    label { display: block; font-size: 0.82rem; color: var(--muted); margin-bottom: 6px; }
    input[type="text"], input[type="password"], textarea {
      width: 100%;
      padding: 12px;
      border-radius: 10px;
      border: 1px solid rgba(148,163,184,0.25);
      background: #0b1222;
      color: var(--text);
      font-size: 16px;
      min-height: var(--touch);
    }
    textarea { min-height: 88px; resize: vertical; }
    input[type="file"] {
      width: 100%;
      font-size: 0.9rem;
      color: var(--muted);
    }
    .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .row.spread { justify-content: space-between; }
    button, .btn {
      min-height: var(--touch);
      padding: 0 14px;
      border-radius: 10px;
      border: none;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      background: var(--accent);
      color: #0f172a;
      flex: 1 1 auto;
      touch-action: manipulation;
    }
    button.secondary { background: #334155; color: var(--text); }
    button.danger { background: #7f1d1d; color: #fecaca; }
    button.ghost { background: transparent; border: 1px solid rgba(148,163,184,0.35); color: var(--text); }
    button:active { opacity: 0.88; transform: scale(0.99); }
    .modes {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
      gap: 8px;
    }
    .modes button {
      flex: unset;
      font-size: 0.82rem;
      line-height: 1.2;
      padding: 10px 8px;
      min-height: calc(var(--touch) - 4px);
    }
    .modes button.active { outline: 2px solid var(--accent2); }
    .hint { font-size: 0.85rem; color: var(--muted); margin-top: 8px; }
    .banner {
      padding: 10px 12px;
      border-radius: 10px;
      background: rgba(56,189,248,0.12);
      border: 1px solid rgba(56,189,248,0.35);
      font-size: 0.9rem;
      margin-bottom: 12px;
      white-space: pre-wrap;
    }
    .banner.warn { background: rgba(248,113,113,0.12); border-color: rgba(248,113,113,0.35); }
    .results img {
      max-width: 100%;
      height: auto;
      border-radius: 10px;
      display: block;
      margin: 10px 0;
      border: 1px solid rgba(148,163,184,0.2);
    }
    .cap { font-size: 0.88rem; color: var(--muted); white-space: pre-wrap; margin-bottom: 12px; }
    .admin { border-left: 3px solid var(--accent2); padding-left: 10px; margin-top: 10px; }
    .hidden { display: none !important; }
    @media (min-width: 720px) {
      .wrap { max-width: 920px; margin: 0 auto; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>BioChecks (Luna)</h1>
    <p class="lead">Те же режимы, что в боте: загрузите фото и/или укажите ссылки.</p>

    <div id="loginCard" class="card hidden">
      <label>Логин (как в access_users)</label>
      <input id="loginUser" type="text" autocomplete="username"/>
      <label style="margin-top:10px">Пароль</label>
      <input id="loginPass" type="password" autocomplete="current-password"/>
      <div class="row" style="margin-top:12px">
        <button type="button" id="btnLogin">Войти</button>
      </div>
    </div>

    <div id="mainCard" class="card hidden">
      <div class="row spread">
        <span id="userInfo" class="hint"></span>
        <button type="button" class="secondary" id="btnLogout" style="flex:0 0 auto;min-width:110px">Выйти</button>
      </div>
      <p class="hint" id="flowInfo"></p>

      <label>Режим</label>
      <div class="modes" id="modes"></div>

      <label style="margin-top:12px">Файлы</label>
      <input id="files" type="file" accept="image/*" multiple/>
      <label style="margin-top:12px">Ссылки на изображения (в тексте)</label>
      <textarea id="urls" placeholder="https://…"></textarea>
      <div class="row" style="margin-top:12px">
        <button type="button" id="btnSend">Отправить</button>
        <button type="button" class="secondary" id="btnResetFlow">Сброс</button>
      </div>

      <div id="adminBox" class="admin hidden">
        <strong>Админ</strong>
        <div class="row" style="margin-top:8px">
          <button type="button" class="ghost" id="btnListUsers">Список</button>
        </div>
        <div class="row" style="margin-top:8px">
          <input id="admLogin" type="text" placeholder="логин" style="flex:2"/>
          <button type="button" class="secondary" id="btnIssue" style="flex:1">Выдать пароль</button>
          <button type="button" class="danger" id="btnRevoke" style="flex:1">Отозвать</button>
        </div>
        <pre id="adminOut" class="hint" style="white-space:pre-wrap"></pre>
      </div>
    </div>

    <div id="banners"></div>
    <div class="card results" id="results"></div>
  </div>
  <script>
(() => {
  const MODE_DEFS = [
    ["wait_compare", "1 к 1"],
    ["wait_attributes", "Атрибуты"],
    ["wait_body_attributes", "Атрибуты тела"],
    ["wait_liveness", "Лайфнесс"],
    ["wait_deepfake", "Дипфейк"],
    ["wait_quality", "Качество"],
    ["wait_face_detect", "Детекция лиц"],
    ["wait_body_detect", "Детекция тел"],
    ["wait_crowd_detect", "Толпа"],
    ["wait_image_modification", "Модификация"],
  ];
  let currentFlow = "idle";

  function el(tag, attrs, text) {
    const e = document.createElement(tag);
    if (attrs) Object.entries(attrs).forEach(([k,v]) => e.setAttribute(k,v));
    if (text != null) e.textContent = text;
    return e;
  }

  async function api(path, opt) {
    const r = await fetch(path, Object.assign({credentials: "same-origin"}, opt || {}));
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      let msg = r.statusText || "Ошибка";
      if (typeof data.detail === "string") msg = data.detail;
      else if (Array.isArray(data.detail)) msg = data.detail.map((d) => d.msg || d.type || "").join("; ");
      throw new Error(msg);
    }
    return data;
  }

  function renderModes() {
    const box = document.getElementById("modes");
    box.innerHTML = "";
    const idle = el("button", { type: "button" }, "Меню / сброс режима");
    idle.className = currentFlow === "idle" ? "secondary active" : "secondary";
    idle.addEventListener("click", async () => {
      await api("/api/reset", { method: "POST" });
      currentFlow = "idle";
      renderModes();
      await refresh();
    });
    box.appendChild(idle);
    for (const [fid, label] of MODE_DEFS) {
      const b = el("button", { type: "button" }, label);
      if (currentFlow === fid) b.classList.add("active");
      b.addEventListener("click", async () => {
        await api("/api/flow", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ flow: fid }) });
        currentFlow = fid;
        renderModes();
        await refresh();
      });
      box.appendChild(b);
    }
  }

  function clearBanners() { document.getElementById("banners").innerHTML = ""; }

  function addBanner(text, warn) {
    const d = el("div", { class: warn ? "banner warn" : "banner" }, "");
    d.textContent = text;
    document.getElementById("banners").appendChild(d);
  }

  function renderResult(payload) {
    const root = document.getElementById("results");
    root.innerHTML = "";
    if (!payload || !payload.items) return;
    for (const it of payload.items) {
      if (it.kind === "text") {
        const p = el("div", { class: "banner" }, it.text);
        root.appendChild(p);
      } else if (it.kind === "image") {
        if (it.caption) root.appendChild(el("div", { class: "cap" }, it.caption));
        const img = el("img", { alt: it.filename || "result" });
        img.src = "data:image/jpeg;base64," + it.data_base64;
        root.appendChild(img);
      }
    }
    if (payload.menu_hint) {
      root.appendChild(el("div", { class: "hint" }, payload.menu_hint));
    }
  }

  async function refresh() {
    const me = await api("/api/me");
    const needLogin = me.auth_enabled && !me.authorized;
    document.getElementById("loginCard").classList.toggle("hidden", !needLogin);
    document.getElementById("mainCard").classList.toggle("hidden", needLogin);
    currentFlow = me.flow || "idle";
    renderModes();
    document.getElementById("userInfo").textContent = me.auth_enabled
      ? (me.login ? ("Вы: " + me.login) : "")
      : "Доступ без пароля";
    let fi = "Режим: " + (me.flow === "idle" ? "ожидание (1 фото — выберите режим; 2 фото — сравнение)" : me.flow);
    if (me.flow === "wait_compare") fi += " — загружено " + me.compare_count + "/2";
    document.getElementById("flowInfo").textContent = fi;
    document.getElementById("adminBox").classList.toggle("hidden", !me.is_admin);
    document.getElementById("btnLogout").classList.toggle("hidden", !me.auth_enabled);
  }

  document.getElementById("btnLogin").addEventListener("click", async () => {
    clearBanners();
    const login = document.getElementById("loginUser").value.trim();
    const password = document.getElementById("loginPass").value;
    try {
      await api("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ login, password }),
      });
      document.getElementById("loginPass").value = "";
      await refresh();
    } catch (e) { addBanner(String(e.message), true); }
  });

  document.getElementById("btnLogout").addEventListener("click", async () => {
    await api("/api/logout", { method: "POST" });
    clearBanners();
    document.getElementById("results").innerHTML = "";
    await refresh();
  });

  document.getElementById("btnResetFlow").addEventListener("click", async () => {
    await api("/api/reset", { method: "POST" });
    currentFlow = "idle";
    renderModes();
    await refresh();
  });

  document.getElementById("btnSend").addEventListener("click", async () => {
    clearBanners();
    document.getElementById("results").innerHTML = "";
    const fd = new FormData();
    fd.append("urls_text", document.getElementById("urls").value);
    const inp = document.getElementById("files");
    for (const f of inp.files) fd.append("files", f);
    try {
      const r = await fetch("/api/analyze", { method: "POST", body: fd, credentials: "same-origin" });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Ошибка");
      for (const w of data.warnings || []) {
        if (w.kind === "text") addBanner(w.text, true);
      }
      if (data.result) renderResult(data.result);
      currentFlow = data.flow;
      renderModes();
      document.getElementById("flowInfo").textContent =
        "Режим: " + (data.flow === "idle" ? "ожидание" : data.flow) +
        (data.flow === "wait_compare" ? " — " + data.compare_count + "/2" : "");
    } catch (e) { addBanner(String(e.message), true); }
  });

  document.getElementById("btnListUsers").addEventListener("click", async () => {
    const u = await api("/api/admin/users");
    document.getElementById("adminOut").textContent = (u.users || []).join("\\n") || "(пусто)";
  });
  document.getElementById("btnIssue").addEventListener("click", async () => {
    const login = document.getElementById("admLogin").value.trim();
    const r = await api("/api/admin/issue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ login }),
    });
    document.getElementById("adminOut").textContent = "Выдан пароль для " + r.login + ":\\n" + r.password;
  });
  document.getElementById("btnRevoke").addEventListener("click", async () => {
    const login = document.getElementById("admLogin").value.trim();
    await api("/api/admin/revoke", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ login }),
    });
    document.getElementById("adminOut").textContent = "Отозвано: " + login;
  });

  refresh().catch(e => addBanner(String(e), true));
})();
  </script>
</body>
</html>
"""
