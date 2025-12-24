import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Loader2, AlertCircle, Globe, Search, ExternalLink, MonitorPlay, RefreshCw, Wifi, WifiOff, MousePointer } from 'lucide-react';
import { cn } from '../../lib/utils';
import { defaultNewTabContent } from '../../data/mock';

const BrowserViewport = ({
  url,
  isLoading,
  screenshot,
  error,
  onMouseMove,
  onMouseClick,
  onKeyPress,
  onScroll,
  onType,
  onNavigate,
  sessionId,
  wsConnected,
  onRetry
}) => {
  const viewportRef = useRef(null);
  const [cursorPosition, setCursorPosition] = useState({ x: 0, y: 0 });
  const [isInteracting, setIsInteracting] = useState(false);
  const [viewportSize, setViewportSize] = useState({ width: 1280, height: 720 });
  const [isMouseDown, setIsMouseDown] = useState(false);
  const lastClickRef = useRef({ time: 0, x: 0, y: 0 });

  // Track viewport size for coordinate scaling
  useEffect(() => {
    if (!viewportRef.current) return;
    
    const updateSize = () => {
      if (viewportRef.current) {
        const rect = viewportRef.current.getBoundingClientRect();
        setViewportSize({ width: rect.width, height: rect.height });
      }
    };
    
    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  // Scale coordinates from viewport to browser coordinates (1280x720)
  const scaleCoordinates = useCallback((x, y) => {
    const scaleX = 1280 / viewportSize.width;
    const scaleY = 720 / viewportSize.height;
    return {
      x: Math.round(x * scaleX),
      y: Math.round(y * scaleY)
    };
  }, [viewportSize]);

  const handleMouseMove = useCallback((e) => {
    if (!viewportRef.current) return;
    const rect = viewportRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setCursorPosition({ x, y });
    
    const scaled = scaleCoordinates(x, y);
    onMouseMove?.(scaled.x, scaled.y);
  }, [onMouseMove, scaleCoordinates]);

  const handleClick = useCallback((e) => {
    if (!viewportRef.current) return;
    e.preventDefault();
    const rect = viewportRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const scaled = scaleCoordinates(x, y);
    
    // Detect double-click (within 300ms and 10px)
    const now = Date.now();
    const last = lastClickRef.current;
    const isDoubleClick = (now - last.time < 300) && 
      Math.abs(scaled.x - last.x) < 10 && 
      Math.abs(scaled.y - last.y) < 10;
    
    lastClickRef.current = { time: now, x: scaled.x, y: scaled.y };
    
    // Pass click count for double-click support
    const clickCount = isDoubleClick ? 2 : 1;
    onMouseClick?.(scaled.x, scaled.y, e.button, clickCount);
  }, [onMouseClick, scaleCoordinates]);

  const handleMouseDown = useCallback((e) => {
    if (!viewportRef.current) return;
    e.preventDefault();
    setIsMouseDown(true);
    viewportRef.current.focus();
  }, []);

  const handleMouseUp = useCallback((e) => {
    setIsMouseDown(false);
  }, []);

  const handleContextMenu = useCallback((e) => {
    e.preventDefault();
    if (!viewportRef.current) return;
    const rect = viewportRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const scaled = scaleCoordinates(x, y);
    onMouseClick?.(scaled.x, scaled.y, 2, 1); // Right click
  }, [onMouseClick, scaleCoordinates]);

  const handleKeyDown = useCallback((e) => {
    if (!isInteracting) return;
    
    // Allow default behavior only for system shortcuts
    const isSystemShortcut = (e.ctrlKey || e.metaKey) && ['r', 'R', 't', 'T', 'w', 'W'].includes(e.key);
    if (isSystemShortcut) return;
    
    e.preventDefault();
    e.stopPropagation();
    
    // Map special keys to Playwright key names
    const keyMap = {
      'Enter': 'Enter',
      'Tab': 'Tab',
      'Backspace': 'Backspace',
      'Delete': 'Delete',
      'Escape': 'Escape',
      'ArrowUp': 'ArrowUp',
      'ArrowDown': 'ArrowDown',
      'ArrowLeft': 'ArrowLeft',
      'ArrowRight': 'ArrowRight',
      'Home': 'Home',
      'End': 'End',
      'PageUp': 'PageUp',
      'PageDown': 'PageDown',
      ' ': 'Space',
      'F1': 'F1', 'F2': 'F2', 'F3': 'F3', 'F4': 'F4',
      'F5': 'F5', 'F6': 'F6', 'F7': 'F7', 'F8': 'F8',
      'F9': 'F9', 'F10': 'F10', 'F11': 'F11', 'F12': 'F12',
    };
    
    const modifiers = {
      ctrl: e.ctrlKey,
      alt: e.altKey,
      shift: e.shiftKey,
      meta: e.metaKey
    };
    
    // Check if it's a special key or has modifiers
    if (keyMap[e.key] || e.ctrlKey || e.altKey || e.metaKey) {
      const key = keyMap[e.key] || e.key;
      onKeyPress?.(key, e.keyCode, modifiers);
    } else if (e.key.length === 1) {
      // Single character - use type for better handling
      if (onType) {
        onType(e.key);
      } else {
        onKeyPress?.(e.key, e.keyCode, modifiers);
      }
    } else {
      onKeyPress?.(e.key, e.keyCode, modifiers);
    }
  }, [isInteracting, onKeyPress, onType]);

  const handleWheel = useCallback((e) => {
    e.preventDefault();
    // Normalize scroll values for smoother scrolling
    const deltaX = e.deltaX;
    const deltaY = e.deltaY;
    onScroll?.(deltaX, deltaY);
  }, [onScroll]);

  // Focus management
  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    
    // Prevent context menu inside viewport
    const preventContext = (e) => e.preventDefault();
    viewport.addEventListener('contextmenu', preventContext);
    
    return () => {
      viewport.removeEventListener('contextmenu', preventContext);
    };
  }, []);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    
    viewport.addEventListener('keydown', handleKeyDown);
    return () => viewport.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // New Tab page
  if (url === 'chrome://newtab' || !url) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center bg-gradient-to-b from-zinc-900 via-zinc-900 to-zinc-950 p-8">
        {/* Session status */}
        <div data-testid="newtab-session-status" className="absolute top-4 right-4 flex items-center gap-2 text-xs">
          {sessionId ? (
            <div className="flex items-center gap-2 bg-zinc-800/80 px-3 py-1.5 rounded-full">
              {wsConnected ? (
                <>
                  <Wifi className="w-3 h-3 text-emerald-400" />
                  <span className="text-emerald-400">Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-3 h-3 text-amber-400" />
                  <span className="text-amber-400">Reconnecting...</span>
                </>
              )}
            </div>
          ) : (
            <button
              data-testid="initialize-browser-button"
              onClick={onRetry}
              className="flex items-center gap-2 bg-zinc-800/80 px-3 py-1.5 rounded-full text-zinc-400 hover:text-zinc-200 transition-colors"
            >
              <RefreshCw className="w-3 h-3" />
              <span>Initialize Browser</span>
            </button>
          )}
        </div>

        {/* Logo */}
        <div className="mb-8">
          <div className="w-24 h-24 rounded-full bg-gradient-to-br from-sky-400 via-indigo-500 to-violet-500 flex items-center justify-center shadow-2xl shadow-sky-500/20">
            <Globe className="w-12 h-12 text-white" />
          </div>
        </div>

        {/* Search box */}
        <div className="w-full max-w-2xl mb-12">
          <div 
            data-testid="newtab-search-box"
            onClick={() => document.querySelector('[data-address-input]')?.focus()}
            className="flex items-center gap-4 bg-zinc-800/80 rounded-full px-6 py-4 border border-zinc-700/50 hover:border-zinc-600 transition-all cursor-text shadow-xl"
          >
            <Search className="w-5 h-5 text-zinc-400" />
            <span className="text-zinc-400 text-lg">Search Google or type a URL</span>
          </div>
        </div>

        {/* Shortcuts */}
        <div className="grid grid-cols-4 gap-6">
          {defaultNewTabContent.shortcuts.map((shortcut, index) => (
            <button
              data-testid={`newtab-shortcut-${shortcut.name.toLowerCase().replace(/\s+/g, '-')}`}
              key={index}
              onClick={() => onNavigate(shortcut.url)}
              className="group flex flex-col items-center gap-3 p-4 rounded-2xl hover:bg-zinc-800/50 transition-all duration-200"
            >
              <div className="w-14 h-14 rounded-2xl bg-zinc-800 flex items-center justify-center group-hover:bg-zinc-700 transition-colors shadow-lg">
                {shortcut.icon === 'search' && <Search className="w-6 h-6 text-sky-400" />}
                {shortcut.icon === 'play' && <MonitorPlay className="w-6 h-6 text-red-400" />}
                {shortcut.icon === 'github' && <Globe className="w-6 h-6 text-zinc-300" />}
                {shortcut.icon === 'message-circle' && <Globe className="w-6 h-6 text-indigo-400" />}
              </div>
              <span className="text-sm text-zinc-400 group-hover:text-zinc-200 transition-colors">
                {shortcut.name}
              </span>
            </button>
          ))}
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center bg-zinc-900 p-8">
        <AlertCircle className="w-16 h-16 text-red-400 mb-4" />
        <h2 className="text-xl font-semibold text-zinc-200 mb-2">This site can't be reached</h2>
        <p className="text-zinc-400 text-center max-w-md mb-6">{error}</p>
        <div className="flex gap-3">
          <button
            onClick={() => onNavigate(url)}
            className="px-6 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg transition-colors"
          >
            Try again
          </button>
          <button
            onClick={() => onNavigate('chrome://newtab')}
            className="px-6 py-2 bg-sky-600 hover:bg-sky-500 text-white rounded-lg transition-colors"
          >
            Go Home
          </button>
        </div>
      </div>
    );
  }

  // Loading state (no screenshot yet)
  if (isLoading && !screenshot) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center bg-zinc-900">
        <Loader2 className="w-12 h-12 text-sky-400 animate-spin mb-4" />
        <p className="text-zinc-400">Loading {url}...</p>
        <p className="text-zinc-500 text-sm mt-2">Rendering with Chromium...</p>
      </div>
    );
  }

  // Browser viewport with screenshot
  return (
    <div
      data-testid="browser-viewport"
      ref={viewportRef}
      className={cn(
        'flex-1 relative overflow-hidden bg-zinc-950 outline-none',
        isInteracting && 'cursor-none'
      )}
      onMouseMove={handleMouseMove}
      onClick={handleClick}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onContextMenu={handleContextMenu}
      onWheel={handleWheel}
      onKeyDown={handleKeyDown}
      onMouseEnter={() => {
        setIsInteracting(true);
        viewportRef.current?.focus();
      }}
      onMouseLeave={() => {
        setIsInteracting(false);
        setIsMouseDown(false);
      }}
      tabIndex={0}
    >
      {screenshot ? (
        <img
          src={screenshot}
          alt="Browser viewport"
          className="w-full h-full object-contain"
          draggable={false}
          style={{ imageRendering: 'auto' }}
        />
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center h-full">
          <Globe className="w-12 h-12 text-zinc-600 mb-4" />
          <p className="text-zinc-500">Waiting for page to render...</p>
          {!wsConnected && sessionId && (
            <button
              onClick={onRetry}
              className="mt-4 flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              Reconnect
            </button>
          )}
        </div>
      )}

      {/* Loading overlay */}
      {isLoading && screenshot && (
        <div className="absolute inset-0 bg-zinc-900/50 flex items-center justify-center">
          <div className="flex items-center gap-3 bg-zinc-800 px-4 py-2 rounded-lg shadow-xl">
            <Loader2 className="w-5 h-5 text-sky-400 animate-spin" />
            <span className="text-zinc-300 text-sm">Loading...</span>
          </div>
        </div>
      )}

      {/* Custom cursor indicator */}
      {isInteracting && (
        <div
          className="absolute w-4 h-4 border-2 border-sky-400 rounded-full pointer-events-none transform -translate-x-1/2 -translate-y-1/2 shadow-lg shadow-sky-400/50"
          style={{ left: cursorPosition.x, top: cursorPosition.y }}
        />
      )}

      {/* Session indicator */}
      {sessionId && (
        <div className="absolute bottom-4 right-4 flex items-center gap-2 bg-zinc-800/90 px-3 py-1.5 rounded-full text-xs">
          <div className={cn(
            "w-2 h-2 rounded-full",
            wsConnected ? "bg-emerald-400 animate-pulse" : "bg-amber-400"
          )} />
          <span className="text-zinc-400">
            {wsConnected ? `Live: ${sessionId.slice(0, 8)}...` : 'Connecting...'}
          </span>
        </div>
      )}

      {/* Connection status overlay */}
      {!wsConnected && sessionId && (
        <div className="absolute top-4 left-4 flex items-center gap-2 bg-amber-500/20 border border-amber-500/30 px-3 py-1.5 rounded-lg text-xs">
          <WifiOff className="w-3 h-3 text-amber-400" />
          <span className="text-amber-300">WebSocket disconnected - using polling</span>
        </div>
      )}
    </div>
  );
};

export default BrowserViewport;
