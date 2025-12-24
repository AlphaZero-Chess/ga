// Browser API Service - Real backend integration
// NOTE (hardening): we intentionally do NOT fall back to a hardcoded '/api'.
// All requests must use REACT_APP_BACKEND_URL.
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const requireBackendUrl = () => {
  if (!BACKEND_URL) {
    // Keep UI running, but make API calls fail loudly and consistently.
    throw new Error(
      "Missing REACT_APP_BACKEND_URL. Set it in your environment (must include '/api' prefix).",
    );
  }
  return BACKEND_URL;
};

const joinUrl = (base, path) => {
  // base can be '/api' or 'https://host/api'.
  const u = new URL(base, window.location.origin);
  const basePath = u.pathname.replace(/\/+$/, "");
  const addPath = path.replace(/^\/+/, "");
  u.pathname = `${basePath}/${addPath}`;
  return u.toString();
};

// Helper to get WebSocket URL for browser streaming.
// We derive WS scheme/host AND preserve the backend base path.
const getWsUrl = () => {
  const apiBase = requireBackendUrl();
  const apiUrl = new URL(apiBase, window.location.origin);
  const wsProtocol = apiUrl.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${wsProtocol}//${apiUrl.host}${apiUrl.pathname.replace(/\/+$/, "")}`;
};

// Browser Session APIs
export const browserApi = {
  // Create a new browser session
  createSession: async () => {
    const response = await fetch(joinUrl(requireBackendUrl(), '/browser/session'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to create session');
    }
    return response.json();
  },

  // Close a browser session
  closeSession: async (sessionId) => {
    const response = await fetch(joinUrl(requireBackendUrl(), `/browser/session/${sessionId}`), {
      method: 'DELETE',
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to close session');
    }
    return response.json();
  },

  // Get session status
  getSessionStatus: async (sessionId) => {
    const response = await fetch(joinUrl(requireBackendUrl(), `/browser/session/${sessionId}/status`));
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to get session status');
    }
    return response.json();
  },

  // Navigate to URL
  navigate: async (sessionId, url) => {
    const response = await fetch(joinUrl(requireBackendUrl(), `/browser/${sessionId}/navigate`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Navigation failed');
    }
    return response.json();
  },

  // Go back
  goBack: async (sessionId) => {
    const response = await fetch(joinUrl(requireBackendUrl(), `/browser/${sessionId}/back`), {
      method: 'POST',
    });
    return response.json();
  },

  // Go forward
  goForward: async (sessionId) => {
    const response = await fetch(joinUrl(requireBackendUrl(), `/browser/${sessionId}/forward`), {
      method: 'POST',
    });
    return response.json();
  },

  // Refresh page
  refresh: async (sessionId) => {
    const response = await fetch(joinUrl(requireBackendUrl(), `/browser/${sessionId}/refresh`), {
      method: 'POST',
    });
    return response.json();
  },

  // Get screenshot
  getScreenshot: async (sessionId) => {
    const response = await fetch(joinUrl(requireBackendUrl(), `/browser/${sessionId}/screenshot`));
    if (!response.ok) {
      throw new Error('Failed to get screenshot');
    }
    return response.json();
  },

  // Click
  click: async (sessionId, x, y, button = 'left') => {
    const response = await fetch(joinUrl(requireBackendUrl(), `/browser/${sessionId}/click`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x, y, button }),
    });
    return response.json();
  },

  // Type text
  type: async (sessionId, text) => {
    const response = await fetch(joinUrl(requireBackendUrl(), `/browser/${sessionId}/type`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    return response.json();
  },

  // Key press
  keypress: async (sessionId, key, modifiers = {}) => {
    const response = await fetch(joinUrl(requireBackendUrl(), `/browser/${sessionId}/keypress`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key, modifiers }),
    });
    return response.json();
  },

  // Scroll
  scroll: async (sessionId, deltaX, deltaY) => {
    const response = await fetch(joinUrl(requireBackendUrl(), `/browser/${sessionId}/scroll`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ delta_x: deltaX, delta_y: deltaY }),
    });
    return response.json();
  },

  // Create WebSocket connection
  createWebSocket: (sessionId) => {
    // wsBaseUrl already includes the same base path as the REST API (e.g. '/api').
    const wsBaseUrl = getWsUrl();
    return new WebSocket(`${wsBaseUrl}/browser/ws/${sessionId}`);
  },
};

// Extensions APIs
export const extensionsApi = {
  // List all extensions
  listExtensions: async () => {
    const response = await fetch(joinUrl(requireBackendUrl(), '/extensions'));
    if (!response.ok) {
      throw new Error('Failed to fetch extensions');
    }
    return response.json();
  },

  // Load unpacked extension
  loadUnpacked: async (path) => {
    const response = await fetch(joinUrl(requireBackendUrl(), '/extensions/load'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to load extension');
    }
    return response.json();
  },

  // Pack extension
  packExtension: async (path, keyPath = null) => {
    const response = await fetch(joinUrl(requireBackendUrl(), '/extensions/pack'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, key_path: keyPath }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to pack extension');
    }
    return response.json();
  },

  // Toggle extension
  toggleExtension: async (extId, enabled) => {
    const response = await fetch(joinUrl(requireBackendUrl(), `/extensions/${extId}/toggle`), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    if (!response.ok) {
      throw new Error('Failed to toggle extension');
    }
    return response.json();
  },

  // Remove extension
  removeExtension: async (extId) => {
    const response = await fetch(joinUrl(requireBackendUrl(), `/extensions/${extId}`), {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error('Failed to remove extension');
    }
    return response.json();
  },
};

// Search APIs
export const searchApi = {
  // Get AI-powered search suggestions
  getSuggestions: async (query, limit = 5) => {
    if (!query || query.length < 2) {
      return { suggestions: [], query };
    }
    
    const response = await fetch(
      joinUrl(requireBackendUrl(), `/search/suggestions?q=${encodeURIComponent(query)}&limit=${limit}`)
    );
    if (!response.ok) {
      return { suggestions: [], query };
    }
    return response.json();
  },
};

export default { browserApi, extensionsApi, searchApi };
