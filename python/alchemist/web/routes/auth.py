"""
认证路由：登录、退出
"""

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from web.auth import COOKIE_NAME, BAN_TIERS

router = APIRouter()


def _next_ban_threshold(fail_count: int) -> int:
    """计算距离下一次封禁还有几次机会"""
    for threshold, _ in BAN_TIERS:
        if fail_count < threshold:
            return threshold - fail_count
    return 0


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """渲染登录页面"""
    templates = request.app.state.templates
    auth: "AuthManager" = request.app.state.auth_manager
    ip = request.client.host

    context = {
        "request": request,
        "error": None,
        "banned": False,
        "ban_remaining": 0,
        "fail_count": 0,
        "next_ban_at": 0,
    }

    banned, remaining = auth.ban_manager.is_banned(ip)
    if banned:
        context["banned"] = True
        context["ban_remaining"] = max(1, remaining // 60)
    else:
        fail_count = auth.ban_manager.get_recent_fail_count(ip)
        context["fail_count"] = fail_count
        context["next_ban_at"] = _next_ban_threshold(fail_count)

    return templates.TemplateResponse("login.html", context)


@router.post("/login")
async def login_submit(request: Request, password: str = Form(...)):
    """处理登录表单提交"""
    templates = request.app.state.templates
    auth: "AuthManager" = request.app.state.auth_manager
    ip = request.client.host

    # 检查 IP 是否被封禁
    banned, remaining = auth.ban_manager.is_banned(ip)
    if banned:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": None,
            "banned": True,
            "ban_remaining": max(1, remaining // 60),
            "fail_count": 0,
            "next_ban_at": 0,
        })

    # 检查频率限制
    if auth.ban_manager.is_rate_limited(ip):
        fail_count = auth.ban_manager.get_recent_fail_count(ip)
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "请求过于频繁，请稍后再试",
            "banned": False,
            "ban_remaining": 0,
            "fail_count": fail_count,
            "next_ban_at": _next_ban_threshold(fail_count),
        })

    # 验证密码
    if auth.verify_password(password):
        # 登录成功
        auth.ban_manager.clear_attempts(ip)
        token = auth.create_session_token()
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(
            key=COOKIE_NAME,
            value=token,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=24 * 60 * 60,
        )
        return response

    # 密码错误
    auth.ban_manager.record_failed_attempt(ip)
    fail_count = auth.ban_manager.get_recent_fail_count(ip)

    # 封禁检查（刚触发的封禁）
    banned, remaining = auth.ban_manager.is_banned(ip)
    if banned:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "密码错误次数过多",
            "banned": True,
            "ban_remaining": max(1, remaining // 60),
            "fail_count": 0,
            "next_ban_at": 0,
        })

    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": "密码错误",
        "banned": False,
        "ban_remaining": 0,
        "fail_count": fail_count,
        "next_ban_at": _next_ban_threshold(fail_count),
    })


@router.get("/logout")
async def logout():
    """退出登录"""
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key=COOKIE_NAME)
    return response
