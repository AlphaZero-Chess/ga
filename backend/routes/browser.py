from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import base64
import uuid
import logging
import os
from datetime import datetime

# Set Playwright browsers path
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/pw-browsers'

from playwright.async_api import async_playwright, Browser, Page, BrowserContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/browser", tags=["browser"])

# Browser session management
class BrowserSessionManager:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
        self.playwright = None
        self.browser: Optional[Browser] = None
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        if self.playwright is None:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process'
                ]
            )
            logger.info("Playwright browser initialized")
    
    async def create_session(self) -> str:
        await self.initialize()
        
        session_id = str(uuid.uuid4())
        context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = await context.new_page()
        
        self.sessions[session_id] = {
            'context': context,
            'page': page,
            'created_at': datetime.utcnow(),
            'history': [],
            'history_index': -1
        }
        
        logger.info(f"Created browser session: {session_id}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[dict]:
        return self.sessions.get(session_id)
    
    async def close_session(self, session_id: str):
        if session_id in self.sessions:
            session = self.sessions[session_id]
            await session['context'].close()
            del self.sessions[session_id]
            logger.info(f"Closed browser session: {session_id}")
    
    async def cleanup(self):
        for session_id in list(self.sessions.keys()):
            await self.close_session(session_id)
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

session_manager = BrowserSessionManager()

# Request/Response models
class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: datetime

class NavigateRequest(BaseModel):
    url: str

class ClickRequest(BaseModel):
    x: float
    y: float
    button: str = "left"

class TypeRequest(BaseModel):
    text: str

class KeyPressRequest(BaseModel):
    key: str
    modifiers: Optional[Dict[str, bool]] = None

class ScrollRequest(BaseModel):
    delta_x: float = 0
    delta_y: float = 0

class SessionStatusResponse(BaseModel):
    session_id: str
    current_url: str
    title: str
    can_go_back: bool
    can_go_forward: bool

class ScreenshotResponse(BaseModel):
    screenshot: str  # base64 encoded
    url: str
    title: str

# Endpoints
@router.post("/session", response_model=CreateSessionResponse)
async def create_session():
    """Create a new browser session"""
    try:
        session_id = await session_manager.create_session()
        session = await session_manager.get_session(session_id)
        return CreateSessionResponse(
            session_id=session_id,
            created_at=session['created_at']
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/session/{session_id}")
async def close_session(session_id: str):
    """Close a browser session"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    await session_manager.close_session(session_id)
    return {"status": "closed"}

@router.get("/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """Get current session status"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    page: Page = session['page']
    history = session['history']
    history_index = session['history_index']
    
    return SessionStatusResponse(
        session_id=session_id,
        current_url=page.url,
        title=await page.title(),
        can_go_back=history_index > 0,
        can_go_forward=history_index < len(history) - 1
    )

@router.post("/{session_id}/navigate")
async def navigate(session_id: str, request: NavigateRequest):
    """Navigate to a URL"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    page: Page = session['page']
    
    try:
        # Use 'domcontentloaded' for proper page render on heavy sites like YouTube
        # 'commit' is too early and causes infinite loading appearance
        await page.goto(request.url, wait_until='domcontentloaded', timeout=60000)
        
        # Wait for network to settle (important for YouTube and dynamic sites)
        try:
            await page.wait_for_load_state('networkidle', timeout=10000)
        except Exception:
            # Fallback - some sites never reach networkidle
            await asyncio.sleep(1)
        
        # Update history
        session['history'] = session['history'][:session['history_index'] + 1]
        session['history'].append(request.url)
        session['history_index'] = len(session['history']) - 1
        
        return {
            "status": "navigated",
            "url": page.url,
            "title": await page.title()
        }
    except Exception as e:
        logger.error(f"Navigation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{session_id}/back")
async def go_back(session_id: str):
    """Go back in history"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session['history_index'] > 0:
        session['history_index'] -= 1
        page: Page = session['page']
        await page.go_back()
        return {"status": "success", "url": page.url}
    
    return {"status": "no_history"}

@router.post("/{session_id}/forward")
async def go_forward(session_id: str):
    """Go forward in history"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session['history_index'] < len(session['history']) - 1:
        session['history_index'] += 1
        page: Page = session['page']
        await page.go_forward()
        return {"status": "success", "url": page.url}
    
    return {"status": "no_forward_history"}

@router.post("/{session_id}/refresh")
async def refresh(session_id: str):
    """Refresh the page"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    page: Page = session['page']
    await page.reload()
    return {"status": "refreshed", "url": page.url}

@router.get("/{session_id}/screenshot", response_model=ScreenshotResponse)
async def get_screenshot(session_id: str):
    """Get current page screenshot"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    page: Page = session['page']
    
    try:
        screenshot_bytes = await page.screenshot(type='jpeg', quality=60)
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        return ScreenshotResponse(
            screenshot=f"data:image/jpeg;base64,{screenshot_base64}",
            url=page.url,
            title=await page.title()
        )
    except Exception as e:
        logger.error(f"Screenshot failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{session_id}/click")
async def click(session_id: str, request: ClickRequest):
    """Click at position"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    page: Page = session['page']
    await page.mouse.click(request.x, request.y, button=request.button)
    return {"status": "clicked"}

@router.post("/{session_id}/type")
async def type_text(session_id: str, request: TypeRequest):
    """Type text"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    page: Page = session['page']
    await page.keyboard.type(request.text)
    return {"status": "typed"}

@router.post("/{session_id}/keypress")
async def keypress(session_id: str, request: KeyPressRequest):
    """Press a key"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    page: Page = session['page']
    
    modifiers = request.modifiers or {}
    keys = []
    
    if modifiers.get('ctrl'):
        keys.append('Control')
    if modifiers.get('alt'):
        keys.append('Alt')
    if modifiers.get('shift'):
        keys.append('Shift')
    if modifiers.get('meta'):
        keys.append('Meta')
    
    keys.append(request.key)
    await page.keyboard.press('+'.join(keys))
    
    return {"status": "pressed"}

@router.post("/{session_id}/scroll")
async def scroll(session_id: str, request: ScrollRequest):
    """Scroll the page"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    page: Page = session['page']
    await page.mouse.wheel(request.delta_x, request.delta_y)
    return {"status": "scrolled"}

# WebSocket for real-time streaming
@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    session = await session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    page: Page = session['page']
    streaming = True
    is_navigating = False
    
    async def stream_screenshots():
        nonlocal streaming, is_navigating
        while streaming:
            try:
                # Skip rapid screenshots during navigation to avoid partial renders
                if is_navigating:
                    await asyncio.sleep(0.5)
                    continue
                    
                screenshot_bytes = await page.screenshot(type='jpeg', quality=50)
                screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                
                await websocket.send_json({
                    "type": "screenshot",
                    "data": f"data:image/jpeg;base64,{screenshot_base64}",
                    "url": page.url,
                    "title": await page.title()
                })
                
                await asyncio.sleep(0.15)  # ~7 FPS for smoother experience
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                await asyncio.sleep(0.5)  # Backoff on error
    
    stream_task = asyncio.create_task(stream_screenshots())
    
    try:
        while True:
            data = await websocket.receive_json()
            event_type = data.get('type')
            
            if event_type == 'navigate':
                url = data.get('url')
                if url:
                    try:
                        is_navigating = True
                        # Use 'domcontentloaded' for proper page render on heavy sites like YouTube
                        await page.goto(url, wait_until='domcontentloaded', timeout=60000)
                        # Wait for network to settle
                        try:
                            await page.wait_for_load_state('networkidle', timeout=10000)
                        except Exception:
                            await asyncio.sleep(1)
                        session['history'].append(url)
                        session['history_index'] = len(session['history']) - 1
                    except Exception as e:
                        await websocket.send_json({"type": "error", "message": str(e)})
                    finally:
                        is_navigating = False
            
            elif event_type == 'click':
                x, y = data.get('x', 0), data.get('y', 0)
                button = data.get('button', 'left')
                await page.mouse.click(x, y, button=button)
            
            elif event_type == 'type':
                text = data.get('text', '')
                await page.keyboard.type(text)
            
            elif event_type == 'keypress':
                key = data.get('key', '')
                await page.keyboard.press(key)
            
            elif event_type == 'scroll':
                delta_x = data.get('deltaX', 0)
                delta_y = data.get('deltaY', 0)
                await page.mouse.wheel(delta_x, delta_y)
            
            elif event_type == 'back':
                if session['history_index'] > 0:
                    session['history_index'] -= 1
                    await page.go_back()
            
            elif event_type == 'forward':
                if session['history_index'] < len(session['history']) - 1:
                    session['history_index'] += 1
                    await page.go_forward()
            
            elif event_type == 'refresh':
                await page.reload()
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    finally:
        streaming = False
        stream_task.cancel()
        try:
            await stream_task
        except asyncio.CancelledError:
            pass

# Cleanup on shutdown
async def cleanup_sessions():
    await session_manager.cleanup()
